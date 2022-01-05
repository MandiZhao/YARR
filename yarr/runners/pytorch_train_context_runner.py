import copy
import logging
import os
from random import sample
import shutil
import signal
import sys
import threading
import time
# from torch.multiprocessing import Lock, cpu_count
from multiprocessing import Lock, cpu_count
from typing import Optional, List
from typing import Union
from collections import defaultdict
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.utils.multitask_rollout_generator import NOISY_VECS, NOISY_VECS_20
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary 
from arm.models.slowfast  import TempResNet 
from omegaconf import DictConfig

from arm.c2farm.context_agent import CONTEXT_KEY # the key context agent looks for in replay samples
from functools import partial 
from einops import rearrange

NUM_WEIGHTS_TO_KEEP = 10
TASK_ID='task_id'
VAR_ID='variation_id'
DEMO_KEY='front_rgb' # what to look for in the demo dataset
ONE_HOT_KEY='var_one_hot'
WAIT_WARN=1000
TRAN_WAIT_WARN=500 
class PyTorchTrainContextRunner(TrainRunner):

    def __init__(self,
                 agent: Agent,
                 env_runner: EnvRunner,
                 wrapped_replay_buffer: Union[PyTorchReplayBuffer, List[PyTorchReplayBuffer]],
                 train_device: torch.device, 
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(1e6),
                 eval_episodes: int = 1000, 
                 logdir: str = '/tmp/yarr/logs',
                 log_freq: int = 10,
                 transitions_before_train: int = 1000,
                 weightsdir: str = '/tmp/yarr/weights',
                 save_freq: int = 100,
                 replay_ratio: Optional[float] = None,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 wandb_logging: bool = False,
                 buffers_per_batch: int = -1, # -1 = all
                 num_tasks_per_batch: int = 1, 
                 context_cfg: DictConfig = None,  # may want to more/less frequently update context
                 train_demo_dataset=None,
                 val_demo_dataset=None,
                 context_device=None,
                 no_context: bool = False, 
                 one_hot: bool = False,
                 noisy_one_hot: bool = False, 
                 num_vars: int = 20,
                 update_buffer_prio: bool = True, 
                 offline: bool = False, # i.e. no env runner 
                 eval_only: bool = False, # no agent update
                 task_var_to_replay_idx: dict = {},
                 switch_online_tasks: int = -1, # if > 0: try setting the env runner to focus on only this many tasks
                 dev_cfg=None, 
                 ):
        super(PyTorchTrainContextRunner, self).__init__(
            agent, env_runner, wrapped_replay_buffer,
            stat_accumulator,
            iterations, logdir, log_freq, transitions_before_train, weightsdir,
            save_freq)
        self._wrapped_buffer = wrapped_replay_buffer if isinstance(
            wrapped_replay_buffer, list) else [wrapped_replay_buffer] 
        self._buffers_per_batch = buffers_per_batch if buffers_per_batch > 0 else len(self._wrapped_buffer)
        self._num_tasks_per_batch = num_tasks_per_batch
        self._buffer_sample_rates = [1.0 / len(self._wrapped_buffer) for _ in range(len(wrapped_replay_buffer))]
        self._per_buffer_error = [1.0 for  _ in range(len(wrapped_replay_buffer))]
        self._update_buffer_prio = update_buffer_prio
        logging.info(f'Created a list of prioties for {len(self._wrapped_buffer)} buffers, each batch samples from {self._buffers_per_batch} of them, \
            Updating priorities for choosing buffers? **{self._update_buffer_prio}**')
 
        self._train_device = train_device
        self._context_device = context_device
        self._tensorboard_logging = tensorboard_logging
        #self._csv_logging = csv_logging

        if replay_ratio is not None and replay_ratio < 0:
            raise ValueError("max_replay_ratio must be positive.")
        self._target_replay_ratio = replay_ratio

        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir, tensorboard_logging, csv_logging, wandb_logging)
        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)
        

        self._no_context = no_context
        if no_context:
            logging.warning('TrainRunner is not sampling/updating context')
        self._context_cfg = context_cfg
        self._train_demo_dataset = train_demo_dataset
        self._val_demo_dataset = val_demo_dataset

        self.ctxt_train_iter = iter(train_demo_dataset) if train_demo_dataset is not None else None 
        self.ctxt_val_iter = iter(val_demo_dataset) if val_demo_dataset is not None else None
        self._one_hot = one_hot
        self._noisy_one_hot = noisy_one_hot 
        self._num_vars = num_vars
        self._offline = offline 
        if offline:
            logging.warning('Train Runner is not spinning up any EnvRunner instances')
        self._eval_only = eval_only
        self.eval_episodes = eval_episodes
        if eval_only:
            logging.warning(f'Only EnvRunner is spinning, no agent update happening, evaluating a total of {eval_episodes} episodes')

        self.switch_online_tasks = switch_online_tasks
        self.task_var_to_replay_idx = task_var_to_replay_idx
        self.task_ids = [int(k) for k in task_var_to_replay_idx.keys()]
        assert self._num_tasks_per_batch <= len(self.task_ids), 'Cannot sample more tasks than avaliable'
        self._task_sample_rates = [1/len(self.task_ids) for _ in self.task_ids]
        self.task_vars = [task_var_to_replay_idx[_id] for _id in self.task_ids]
        print('Two-layer sampling from tasks and var ids:', self.task_ids, self.task_vars)
        self.online_task_ids = [int(k) for k in task_var_to_replay_idx.keys()]
        if switch_online_tasks > 0:
            assert switch_online_tasks <= len(self.online_task_ids), f"Cannot select more tasks than avaliable"
            logging.warning(f'Environment runner priority-selects {switch_online_tasks} tasks from a total of {len(self.online_task_ids)} to run')

        self.dev_cfg = dev_cfg 
         

    def _save_model(self, i):
        with self._save_load_lock:
            d = os.path.join(self._weightsdir, str(i))
            os.makedirs(d, exist_ok=True)
            self._agent.save_weights(d)
            # Remove oldest save
            prev_dir = os.path.join(self._weightsdir, str(
                i - self._save_freq * NUM_WEIGHTS_TO_KEEP))
            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)
    
    def _sample_replay(self, data_iter):
        # New: sample fixed number of different tasks
        sampled_buf_ids = []
        if len(data_iter) == 1:
            sampled_buf_ids = [0]
        elif self.dev_cfg.batch_sample_mode == 'equal-task':
            buffs_per_task = int(self._buffers_per_batch / len(self.task_ids)) # 5 
            for task_id in self.task_ids:
                sampled_buf_ids.extend(
                    list(
                        np.random.choice(
                            list(self.task_var_to_replay_idx[task_id].values()),
                            size=buffs_per_task,
                            replace=(True if len(self.task_var_to_replay_idx[task_id].values()) < buffs_per_task else False),
                        )
                    )
                )
        elif self.dev_cfg.batch_sample_mode == 'equal-var':
            assert len(self.task_ids) == 3, 'Only 3 tasks supported for equal-var'
            for task_id in self.task_ids:
                sampled_buf_ids.extend(
                    list(
                        np.random.choice(
                            list(self.task_var_to_replay_idx[task_id].values()),
                            size=(1 if len(self.task_var_to_replay_idx[task_id].values()) == 1 else 13),
                            replace=True,
                        )
                    )
                )     
        elif self._num_tasks_per_batch > 0:
            sampled_task_ids = np.random.choice(
                a=self.task_ids,
                size=self._num_tasks_per_batch, 
                replace=False, 
                p=self._task_sample_rates,
                )
            sampled_buf_ids = [
                np.random.choice(
                    list(self.task_var_to_replay_idx[task_id].values()), 
                    size=1)[0] for task_id in sampled_task_ids]
            # while len(sampled_buf_ids) < self._buffers_per_batch:
            #     task_id = np.random.choice(
            #         sampled_task_ids, size=1)[0]
            #     sampled_buf_ids.append(
            #         np.random.choice(
            #             list(self.task_var_to_replay_idx[task_id].values()), 
            #             size=1)[0]
            #     )
            other_task_ids = np.random.choice(
                a=self.task_ids,
                size=int(self._buffers_per_batch - len(sampled_buf_ids)),
                replace=True, 
                p=self._task_sample_rates,
                )
            sampled_buf_ids.extend(
                [np.random.choice(
                    list(self.task_var_to_replay_idx[task_id].values()), size=1)[0] for task_id in other_task_ids ])
            

            # print('sampled task ids:', sampled_task_ids, sampled_buf_ids)
        else:
            sampled_buf_ids = np.random.choice(
                a=range(len(data_iter)), 
                size=self._buffers_per_batch, 
                replace=False, 
                p=self._buffer_sample_rates
                )
            # NOTE: WITH replacement for now 
            # np.random.choice(range(len(datasets)), self._buffers_per_batch, replace=False)
            # print('SAMPLED IDS', sampled_buf_ids)
 
        sampled_batch = []
        for j in sampled_buf_ids:
            one_buf = next(data_iter[j]) 
            one_buf['buffer_id'] = torch.tensor(
                [j for _ in range(self._wrapped_buffer[j].replay_buffer.batch_size) ], dtype=torch.int64)
            if not self._no_context:
                task_ids, variation_ids = one_buf[TASK_ID], one_buf[VAR_ID]
                task, var = task_ids[0], variation_ids[0]
                assert torch.all(task_ids == task) and torch.all(variation_ids == var), f'For now, assume each buffer contains only 1 vairation from 1 task'
                var_ids_tensor = variation_ids.clone().detach().to(torch.int64)
                if self._one_hot: 
                    # demo_samples = F.one_hot(var_ids_tensor, num_classes=self._num_vars)
                    demo_samples = F.one_hot(one_buf['buffer_id'], num_classes=len(data_iter))
                    one_buf[CONTEXT_KEY] = demo_samples.clone().detach().to(torch.float32) 
                elif self._noisy_one_hot: 
                    demo_samples = torch.stack( [
                        torch.tensor(NOISY_VECS[int(_id)]) for _id in variation_ids], 0).to(torch.float32)
                    one_buf[CONTEXT_KEY] = demo_samples 
                elif self.dev_cfg.get('noisy_dim_20', False):
                    demo_samples = torch.stack([
                            torch.tensor(NOISY_VECS_20[int(_id)]) for _id in variation_ids], 0).to(torch.float32)
                    one_buf[CONTEXT_KEY] = demo_samples 
                else:
                    # demo_samples = self._train_demo_dataset.sample_for_replay(task_ids, variation_ids) # -> this matches every single variation to a context video (B,K,...) -> (B,N,T,3,128,128)
                    #TODO(1014): change here to no match, action samples either use single or randomly sample from context samples
                    demo_samples = self._train_demo_dataset.sample_for_replay_no_match(task, var) # draw K independent samples 
                    one_buf[CONTEXT_KEY] = torch.stack( [ d[DEMO_KEY] for d in demo_samples ], dim=0)
                    # print(f'task {task} var {var}, context sample: ', one_buf[CONTEXT_KEY].shape)
                    #print(one_buf[CONTEXT_KEY].shape) # should be (num_sample, video_len, 3, 128, 128) 
                    one_buf[ONE_HOT_KEY] = F.one_hot(var_ids_tensor, num_classes=self._num_vars).detach().to(torch.float32) 
            sampled_batch.append(one_buf)

        result = {}
        for key in sampled_batch[0]:
            result[key] = torch.stack([d[key] for d in sampled_batch], 0) # shape (num_buffer, num_sample, ...
            # result[key] = torch.cat([d[key] for d in sampled_batch], 0)
            # print('context runner:', sampled_batch[0][key].shape, result[key].shape )
        sampled_batch = result
        
        return sampled_batch, sampled_buf_ids 

    def _step(self, i, sampled_batch, buffer_ids):
        update_dict = self._agent.update(i, sampled_batch)
        # new version: use mask select
        prio_key = 'priority' 
        priority = update_dict[prio_key] 
        # task_prio = update_dict['task_prio'] 
        # buff_priority = update_dict['var_prio'] 
        indices = rearrange(sampled_batch['indices'], 'b k ... -> (b k) ... ')
        # print('context runner: returned priorities', priority.shape)
        sampled_buffer_ids = rearrange(sampled_batch['buffer_id'], 'b k ... -> (b k) ... ')
        new_buff_prio = []
        for buf_id in buffer_ids:
            buf_mask = sampled_buffer_ids == buf_id 
            indices_ = torch.masked_select(indices, buf_mask)
            priority_ = torch.masked_select(priority, buf_mask).cpu().detach().numpy() 
            max_prio = self._wrapped_buffer[buf_id].replay_buffer.get_max_priority() # swap NaN with max priority 
            priority_ = np.nan_to_num(priority_, copy=True, nan=max_prio)
            
            self._wrapped_buffer[buf_id].replay_buffer.set_priority(
                indices_.cpu().detach().numpy(), 
                priority_)
            
            if len(buffer_ids) >= 1 and self._update_buffer_prio: 
                self._per_buffer_error[buf_id] = self._wrapped_buffer[buf_id].replay_buffer.get_average_priority()  
                # self._per_buffer_error[buf_id] = (1 - alpha) * self._per_buffer_error[buf_id] + alpha * buffer_prio[0].cpu().detach().item()
        
        # if i % 20 == 0:
        #     print(f'Buffer prio avg: {[self._wrapped_buffer[buf_id].replay_buffer.get_average_priority() for buf_id in range(10)]}')

        if self._update_buffer_prio:
            sum_error = sum(self._per_buffer_error)
            self._buffer_sample_rates = [ e/sum_error for e in self._per_buffer_error]
             
            # print('updated buffer sample rates:', self._buffer_sample_rates) 
        per_task_errors = dict()
        for task_id, v in self.task_var_to_replay_idx.items():
            per_task_errors[task_id] = np.mean([
                self._per_buffer_error[buff_idx] for var_id, buff_idx in v.items()])
        # task_high_to_low = [pair[0] for pair in sorted(per_task_errors.items(), key=lambda item: -item[1])] 
        per_task_errors = sorted(per_task_errors.items(), key=lambda item: item[0]) # list of pairs: [ (task_id, error), ... ]
        sum_error = sum([pair[1] for pair in per_task_errors])
        self._task_sample_rates = [ e/sum_error for e in [pair[1] for pair in per_task_errors]]

        if self.switch_online_tasks > 0:
            
            self.online_task_ids = list(np.random.choice(
                a=range(len(per_task_errors)), 
                size=self.switch_online_tasks,
                replace=False, 
                p=self._task_sample_rates
                ))
        # prio_key = 'priority' 
        # priority = update_dict[prio_key].cpu().detach().numpy() if isinstance(update_dict[prio_key], torch.Tensor) \
        #     else np.numpy(update_dict[prio_key])
        # indices  = sampled_batch['indices'].cpu().detach().numpy()
        # acc_bs = 0
        # for wb_idx, wb in enumerate(self._wrapped_buffer):
        #     bs = wb.replay_buffer.batch_size
        #     if 'priority' in update_dict:
        #         indices_ = indices[:, wb_idx]
        #         if len(priority.shape) > 1:
        #             priority_ = priority[:, wb_idx]
        #         else:
        #             # legacy version
        #             priority_ = priority[acc_bs: acc_bs + bs]
        #         wb.replay_buffer.set_priority(indices_, priority_)
        #     acc_bs += bs

    def _signal_handler(self, sig, frame):
        if threading.current_thread().name != 'MainThread':
            return
        logging.info('SIGINT captured. Shutting down.'
                     'This may take a few seconds.')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]
        sys.exit(0)

    def _get_add_counts(self):
        return np.array([
            r.replay_buffer.add_count for r in self._wrapped_buffer])

    def _get_sum_add_counts(self, avg=False):
        sums = sum([
            r.replay_buffer.add_count for r in self._wrapped_buffer])
        # print('current add sums for all buffer: ', sums)
        if avg:
            return np.mean(sums)
        return sums 

    def _get_min_add_counts(self):
        return np.min([
            r.replay_buffer.add_count for r in self._wrapped_buffer])

    def start(self, resume_dir: str = None):

        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock() 
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device, context_device=self._context_device)
        if resume_dir is not None:
            logging.info('Resuming from checkpoint weights AND saving to a new step-0 for env workers to load')
            print(resume_dir)
            self._agent.load_weights(resume_dir)

        if self._weightsdir is not None:
            self._save_model(0)  # Save weights so workers can load.

        # init_replay_size = self._get_sum_add_counts().astype(float)
        # NOTE(1018): with multiple buffers this intial size was summed to really large, so the env runner struggled to catch up 
        init_replay_size = np.mean(self._get_add_counts()).astype(float) # try setting this to average across buffers

        logging.info('Need %d samples before training. Currently have %s in each buffer, which adds to %d in total, setting init_replay_size to: %s' %
                (self._transitions_before_train, str(self._get_add_counts()), self._get_sum_add_counts(), init_replay_size)     )
        
        single_buffer_bsize = self._wrapped_buffer[0].replay_buffer.batch_size
         
        # Kick off the environments
        if not self._offline:
            self._env_runner.start(self._save_load_lock)
        if not self._eval_only:
            transition_wait = 0
            while (self._get_sum_add_counts() < self._transitions_before_train or self._get_min_add_counts() < single_buffer_bsize):
                time.sleep(1) 
                transition_wait += 1
                if transition_wait % TRAN_WAIT_WARN == 0:
                    logging.info('Waiting for %d total samples before training. Currently have %s.' %
                    (self._transitions_before_train, str(self._get_sum_add_counts())))

        datasets = [r.dataset() for r in self._wrapped_buffer]
        
        assert np.all(
            np.equal([r.replay_buffer.batch_size for r in self._wrapped_buffer], single_buffer_bsize)), 'The replay buffers should all have the same bath size'
        data_iter = [iter(d) for d in datasets] 
         
        
        batch_times_buffers_per_sample = int(single_buffer_bsize  * self._buffers_per_batch )
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        cstep = 0 # in case we want pre-training
        for cstep in range(self._context_cfg.pretrain_context_steps): 
            context_batch = next(self.ctxt_train_iter)
            context_update_dict = self._agent.update_context(cstep, context_batch) 
            if cstep % self._context_cfg.val_freq == 0:
                self.validate_context(cstep)
            if cstep % self._log_freq == 0:
                agent_summaries = self._agent._context_agent.update_summaries() # only about context losses
                self._writer.log_context_only(cstep, agent_summaries)
        
        # for j in range(10):
        #     sampled_batch, sampled_buf_ids = self._sample_replay(data_iter) 
        #     self._agent.visualize_batch(j, sampled_batch)

        # raise ValueError

        for cstep in range(self._context_cfg.pretrain_replay_steps): 
            sampled_batch, sampled_buf_ids = self._sample_replay(data_iter) 
            train_classifier = self.dev_cfg.get('classify', False)
            if self.dev_cfg.get('freeze_after_steps', -1) > 0 and cstep < self.dev_cfg.get('freeze_after_steps', -1):
                train_classifier = False # not touching the classifier loss before embedding freeze
            replay_update_dict = self._agent.update_context_via_qagent(
                cstep, 
                sampled_batch, 
                classify=train_classifier, 
                emb_weight=self.dev_cfg.get('emb_weight', 1.0)) # returns context_agent._replay_summaries
            if cstep == self.dev_cfg.get('freeze_after_steps', -1):
                assert self.dev_cfg.get('classify', False), 'Should be training classifers after freezing'
                logging.info('Freezing embedding network but keep updating the classifier MLPs !')
                self._agent.rebuild_optimizer()
            if cstep % self._log_freq == 0: 
                logging.info('Logging context update step %d, emb loss: %s, acc: %s' % (
                    cstep, replay_update_dict.get('replay_batch/emb_loss', 0), replay_update_dict.get('replay_batch/emd_acc', 0))
                ) 
                if train_classifier:
                    logging.info('Classification pred loss: %s' % (
                    replay_update_dict['replay_batch/pred_loss'].item())) 
                self._writer.log_context_only(
                    cstep, self._agent._context_agent.update_summaries()) # only about context losses
        if self.dev_cfg.get('freeze_emb', False):
            logging.info('Freezing embedding after context pre-training!')
            self._agent.rebuild_optimizer()
        context_step = 0
        buffer_summaries = defaultdict(list)
        recent_online_task_ids = []
        for i in range(self._iterations):
            self._env_runner.set_step(i)

            log_iteration = i % self._log_freq == 0 or i == self._iterations - 1  

            if log_iteration:
                process.cpu_percent(interval=None)

            def get_replay_ratio():
                if self._offline:
                    return 0 
                if self._eval_only:
                    return 100 
                size_used = batch_times_buffers_per_sample * i
                # size_used = single_buffer_bsize * i
                size_added = (
                    self._get_sum_add_counts(avg=True) - init_replay_size
                )
                replay_ratio = size_used / (size_added + 1e-6)
                return replay_ratio
 
            if self._target_replay_ratio is not None and not self._offline:
                # wait for env_runner collecting enough samples
                slept = 0
                while True: 
                    replay_ratio = get_replay_ratio()
                    self._env_runner.current_replay_ratio.value = replay_ratio
                    if replay_ratio < self._target_replay_ratio:
                        break
                    time.sleep(1)
                    slept += 1
                    if slept % WAIT_WARN == 0:
                        logging.warning('Step %d : Train Runner have been waiting for replay_ratio %f to be less than %f for %s seconds.' %
                            (i, replay_ratio, self._target_replay_ratio, slept)
                            ) 
                del replay_ratio

            t = time.time() 
            if self.switch_online_tasks > 0: # select online tasks! 
                self._env_runner.online_task_ids[:] = self.online_task_ids
                recent_online_task_ids.extend(self.online_task_ids)
            
            sample_time, step_time = 0, 0
            
            if not self._eval_only:
                sampled_batch, sampled_buf_ids = self._sample_replay(data_iter) 
                sample_time = time.time() - t
                # print('context runner buff ids:', result['buffer_id'])
 
                for key in [VAR_ID, TASK_ID, 'buffer_id']: 
                    buffer_summaries[key].extend(
                        list(sampled_batch[key].cpu().detach().numpy().flatten() ))
                    # print(key, buffer_summaries[key])
                t = time.time() 
                self._step(i, sampled_batch, sampled_buf_ids)
                step_time = time.time() - t
   
            if (i > 0) and (not self._no_context) and (not self._one_hot) and (not self._noisy_one_hot) and (i % self._context_cfg.update_freq == 0):
                if context_step % self._context_cfg.val_freq == 0:
                        self.validate_context(context_step)
                for _ in range(self._context_cfg.num_update_itrs): 
                    context_batch = next(self.ctxt_train_iter)
                    context_update_dict = self._agent.update_context(i, context_batch)
                    context_step += 1
                    
                    if context_step % self._log_freq == 0:
                        agent_summaries = self._agent.update_summaries() # should be context losses
                        self._writer.log_context_only(context_step, agent_summaries)

            if log_iteration and self._writer is not None:
                replay_ratio = get_replay_ratio()
                if not self._eval_only:
                    logging.info('Step %d. Sample time: %s. Step time: %s. Replay ratio: %s.' % (
                             i, sample_time, step_time, replay_ratio))
                agent_summaries = []
                if not self._eval_only:
                    agent_summaries = self._agent.update_summaries()
                env_summaries = self._env_runner.summaries()
                if self.switch_online_tasks > 0: 
                    env_summaries += [HistogramSummary('online task ids', recent_online_task_ids)] 
                    recent_online_task_ids = []

                buffer_histograms = [] 
                if len(buffer_summaries) > 0:
                    buffer_histograms = [
                        HistogramSummary(key, val) for key, val in buffer_summaries.items()]
                    buffer_summaries = defaultdict(list) # clear 
                self._writer.add_summaries(i, agent_summaries + env_summaries + buffer_histograms)

                # disable all buffer logging for now 
                # for r_i, wrapped_buffer in enumerate(self._wrapped_buffer):
                #     self._writer.add_scalar(
                #         i, 'replay%d/add_count' % r_i,
                #         wrapped_buffer.replay_buffer.add_count)
                #     self._writer.add_scalar(
                #         i, 'replay%d/size' % r_i,
                #         wrapped_buffer.replay_buffer.replay_capacity
                #         if wrapped_buffer.replay_buffer.is_full()
                #         else wrapped_buffer.replay_buffer.add_count)

                self._writer.add_scalar(
                    i, 'replay/replay_ratio', replay_ratio)
                self._writer.add_scalar(
                    i, 'replay/update_to_insert_ratio',
                    float(i) / float(
                        self._get_sum_add_counts() -
                        init_replay_size + 1e-6)) 
                self._writer.add_scalar(
                    i, 'monitoring/sample_time_per_item',
                    sample_time / batch_times_buffers_per_sample)
                self._writer.add_scalar(
                    i, 'monitoring/train_time_per_item',
                    step_time / batch_times_buffers_per_sample)
                self._writer.add_scalar(
                    i, 'monitoring/memory_gb',
                    process.memory_info().rss * 1e-9)
                self._writer.add_scalar(
                    i, 'monitoring/cpu_percent',
                    process.cpu_percent(interval=None) / num_cpu)

            if self._env_runner._iter_eval and self._writer is not None:
                ckpt, summs = self._env_runner.try_log_ckpt_eval()
                if ckpt > -1:
                    assert len(summs) != 0, 'Accumulator is empty!'
                    logging.info(f'Logging all {len(summs)} evaluation data from checkpoint step: {ckpt}')
                    self._writer.log_ckpt_eval(ckpt, summs)
            
            self._writer.end_iteration()
 
            if (i % self._save_freq == 0 or i == self._iterations-1) and self._weightsdir is not None and not self._eval_only:
                self._save_model(i)


        

        logging.info('Stopping envs ...')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]
        logging.info('Stopping log writer')
        if self._writer is not None:
            self._writer.close()

    def evaluate(self, resume_dir: str = None):

        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock() 
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device, context_device=self._context_device)
        if resume_dir is not None:
            logging.info('Resuming from checkpoint weights AND saving to a new step-0 for env workers to load')
            print(resume_dir)
            self._agent.load_weights(resume_dir)

        if self._weightsdir is not None:
            self._save_model(0)  # Save weights so workers can load.
 
        # Kick off the environments
        self._env_runner.start(self._save_load_lock)
        num_episodes = self._env_runner._total_episodes.get('eval_envs', 0)
        last_logged = num_episodes
        while num_episodes <= self.eval_episodes:
            # print(num_episodes)
            time.sleep(1)
            if num_episodes % self._log_freq == 0 and num_episodes != last_logged:
                # self._env_runner.set_step(i)     
                env_summaries = self._env_runner.summaries() 
                self._writer.add_summaries(num_episodes, env_summaries) 
                last_logged = num_episodes
                logging.info(f'logging at {num_episodes} evaled episodes')

            num_episodes = self._env_runner._total_episodes.get('eval_envs', 0)

        self._writer.end_iteration()
 
        if self._writer is not None:
            self._writer.close()

        logging.info('Stopping envs ...')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]

    def validate_context(self, step):
        context_batch = next(self.ctxt_val_iter)
        val_info = self._agent.validate_context(step, context_batch)
        return 


