import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from torch.multiprocessing import Lock, cpu_count
# from multiprocessing import Lock, cpu_count
from typing import Optional, List
from typing import Union

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
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary

from arm.demo_dataset import MultiTaskDemoSampler, RLBenchDemoDataset, collate_by_id
from arm.models.slowfast  import TempResNet
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from arm.c2farm.context_agent import CONTEXT_KEY # the key context agent looks for in replay samples
from functools import partial 

NUM_WEIGHTS_TO_KEEP = 10
TASK_ID='task_id'
VAR_ID='variation_id'
DEMO_KEY='front_rgb' # what to look for in the demo dataset


class PyTorchTrainContextRunner(TrainRunner):

    def __init__(self,
                 agent: Agent,
                 env_runner: EnvRunner,
                 wrapped_replay_buffer: Union[
                     PyTorchReplayBuffer, List[PyTorchReplayBuffer]],
                 train_device: torch.device,
                 replay_buffer_sample_rates: List[float] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(1e6),
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
                 context_cfg: DictConfig = None,  # may want to more/less frequently update context
                 train_demo_dataset=None,
                 val_demo_dataset=None,
                 context_device=None,
                 no_context: bool = False, 
                 one_hot: bool = False,
                 num_vars: int = 20,
                 update_buffer_prio: bool = True, 
                #  ctxt_train_loader = None,
                #  ctxt_val_loader = None,
                 ):
        super(PyTorchTrainContextRunner, self).__init__(
            agent, env_runner, wrapped_replay_buffer,
            stat_accumulator,
            iterations, logdir, log_freq, transitions_before_train, weightsdir,
            save_freq)

        env_runner.log_freq = log_freq
        env_runner.target_replay_ratio = replay_ratio
        self._wrapped_buffer = wrapped_replay_buffer if isinstance(
            wrapped_replay_buffer, list) else [wrapped_replay_buffer]
        self._num_total_buffers = len(self._wrapped_buffer)
        self._buffers_per_batch = buffers_per_batch if buffers_per_batch > 0 else self._num_total_buffers
        self._buffer_sample_rates = [1.0 / self._num_total_buffers for _ in range(len(wrapped_replay_buffer))]
        self._per_buffer_error = [1.0 for  _ in range(len(wrapped_replay_buffer))]
        self._update_buffer_prio = update_buffer_prio
        logging.info(f'Created a list of prioties for {self._num_total_buffers} buffers, each batch samples from {self._buffers_per_batch} of them, \
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
            logging.info('Warning! TrainRunner is not sampling/updating context')
        self._context_cfg = context_cfg
        self._train_demo_dataset = train_demo_dataset
        self._val_demo_dataset = val_demo_dataset

        self.ctxt_train_iter = iter(train_demo_dataset) if train_demo_dataset is not None else None 
        self.ctxt_val_iter = iter(val_demo_dataset) if val_demo_dataset is not None else None
        self._one_hot = one_hot
        self._num_vars = num_vars
         

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
    
    def _step(self, i, sampled_batch, buffer_ids):
        update_dict = self._agent.update(i, sampled_batch)
        # new version: use mask select
        prio_key = 'priority' 
        priority = update_dict[prio_key] 
        buff_priority = update_dict['var_prio'] 
        indices = sampled_batch['indices']
        # print('context runner: returned priorities', priority)
        new_buff_prio = []
        for buf_id in buffer_ids:
            buf_mask = sampled_batch['buffer_id'] == buf_id 
            indices_ = torch.masked_select(indices, buf_mask)
            priority_ = torch.masked_select(priority, buf_mask)
            self._wrapped_buffer[buf_id].replay_buffer.set_priority(
                indices_.cpu().detach().numpy(), 
                priority_.cpu().detach().numpy())
            
            if len(buffer_ids) >= 1 and self._update_buffer_prio:
                # buffer_prio = torch.masked_select(buff_priority, buf_mask) 
                # assert torch.all(buffer_prio[0] == buffer_prio)
                # print(f'Setting error from {self._per_buffer_error[buf_id]} to \
                #     {buffer_prio[0].cpu().detach().item()} ')
                # alpha = 0.7
                # if buf_id == 0:
                #     self._per_buffer_error[buf_id] = 5
                # else:
                self._per_buffer_error[buf_id] = self._wrapped_buffer[buf_id].replay_buffer.get_average_priority()  
                # self._per_buffer_error[buf_id] = (1 - alpha) * self._per_buffer_error[buf_id] + alpha * buffer_prio[0].cpu().detach().item()
        
        # if i % 20 == 0:
        #     print(f'Buffer prio avg: {[self._wrapped_buffer[buf_id].replay_buffer.get_average_priority() for buf_id in range(10)]}')

        if self._update_buffer_prio:
            sum_error = sum(self._per_buffer_error)
            self._buffer_sample_rates = [ e/sum_error for e in self._per_buffer_error]
            # print('updated buffer sample rates:', self._buffer_sample_rates) 

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

    def _get_sum_add_counts(self):
        return sum([
            r.replay_buffer.add_count for r in self._wrapped_buffer])

    def start(self, resume_dir: str = None):

        signal.signal(signal.SIGINT, self._signal_handler)

        self._save_load_lock = Lock()
        
        # Kick off the environments
        self._env_runner.start(self._save_load_lock)

        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device, context_device=self._context_device)
        if resume_dir is not None:
            logging.info('Resuming from checkpoint weights:')
            print(resume_dir)

        if self._weightsdir is not None:
            self._save_model(0)  # Save weights so workers can load.

        logging.info('Need %d samples before training. Currently have %s.' %
                (self._transitions_before_train, str(self._get_add_counts())))
        while (np.any(self._get_sum_add_counts() < self._transitions_before_train)):
            time.sleep(1)
            logging.info(
                'Waiting for %d samples before training. Currently have %s.' %
                (self._transitions_before_train, str(self._get_sum_add_counts())))

        datasets = [r.dataset() for r in self._wrapped_buffer]
        data_iter = [iter(d) for d in datasets] 
         
        init_replay_size = self._get_sum_add_counts().astype(float)
        batch_times_buffers_per_sample = sum([r.replay_buffer.batch_size for r in self._wrapped_buffer[:self._buffers_per_batch]])
        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()

        context_step = 0 
        for _ in range(self._context_cfg.pretrain_context_steps): 
            context_batch = next(self.ctxt_train_iter)
            context_update_dict = self._agent.update_context(context_step, context_batch)
            context_step += 1
            if context_step % self._context_cfg.val_freq == 0:
                self.validate_context(context_step)
            if context_step % self._log_freq == 0:
                agent_summaries = self._agent._context_agent.update_summaries() # only about context losses
                self._writer.log_context_only(context_step, agent_summaries)

        for i in range(self._iterations):
            self._env_runner.set_step(i)

            log_iteration = i % self._log_freq == 0  

            if log_iteration:
                process.cpu_percent(interval=None)

            def get_replay_ratio():
                size_used = batch_times_buffers_per_sample * i
                size_added = (
                    self._get_sum_add_counts()
                    - init_replay_size
                )
                replay_ratio = size_used / (size_added + 1e-6)
                return replay_ratio

            ### comment out for running w/o env runner
            if self._target_replay_ratio is not None:
                # wait for env_runner collecting enough samples
                while True:
                    replay_ratio = get_replay_ratio()
                    self._env_runner.current_replay_ratio.value = replay_ratio
                    if replay_ratio < self._target_replay_ratio:
                        break
                    time.sleep(1)
                    logging.debug(
                        'Waiting for replay_ratio %f to be less than %f.' %
                        (replay_ratio, self._target_replay_ratio))
                del replay_ratio

            t = time.time()

            if len(datasets) == 1:
                sampled_buf_ids = [0] 
            else:
                sampled_buf_ids = np.random.choice(
                    a=range(len(datasets)), 
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
                    [j for _ in range(self._wrapped_buffer[j].replay_buffer.batch_size) ], 
                    dtype=torch.int64)
                sampled_batch.append(one_buf)
            result = {}
            for key in sampled_batch[0]:
                # result[key] = torch.stack([d[key] for d in sampled_batch], 1)
                result[key] = torch.cat([d[key] for d in sampled_batch], 0)
                # print('context runner:', sampled_batch[0][key].shape, result[key].shape )
            sampled_batch = result
            # print('context runner buff ids:', result['buffer_id'])
            buffer_summaries = []
            for key in [VAR_ID, TASK_ID, 'buffer_id']:
                buffer_summaries.append(
                    HistogramSummary(key+'_in_batch', sampled_batch[key].cpu().detach().numpy()
                    )
                )

            if not self._no_context:
                task_ids, variation_ids = sampled_batch[TASK_ID], sampled_batch[VAR_ID]
                # this slows down sampling time ~6x 
                #print('trainer trying to match ids:', task_ids, variation_ids)
                if self._one_hot:
                    var_ids_tensor = variation_ids.clone().detach().to(torch.int64) #torch.tensor(variation_ids, dtype=torch.int64) 
                    demo_samples = F.one_hot(var_ids_tensor, num_classes=self._num_vars)
                    sampled_batch[CONTEXT_KEY] = demo_samples.clone().detach().to(torch.float32) 
                else:
                    demo_samples = self._train_demo_dataset.sample_for_replay(task_ids, variation_ids)
                    sampled_batch[CONTEXT_KEY] = torch.stack(
                        [ d[DEMO_KEY][None] for d in demo_samples ])
                # print(sampled_batch[CONTEXT_KEY].shape) # (bsize, 1, video_len, 3, 128, 128) 
            sample_time = time.time() - t

            batch = {k: v.to(self._train_device) for k, v in sampled_batch.items()}
            t = time.time()
            self._step(i, batch, sampled_buf_ids)
            step_time = time.time() - t
             
            if (not self._no_context) and (not self._one_hot) and (i % self._context_cfg.update_freq == 0):
                for _ in range(self._context_cfg.num_update_itrs): 
                    context_batch = next(self.ctxt_train_iter)
                    context_update_dict = self._agent.update_context(i, context_batch)
                    context_step += 1
                    if context_step % self._context_cfg.val_freq == 0:
                        self.validate_context(context_step)
                    if context_step % self._log_freq == 0:
                        agent_summaries = self._agent.update_summaries() # should be context losses
                        self._writer.log_context_only(context_step, agent_summaries)


            if log_iteration and self._writer is not None:
                replay_ratio = get_replay_ratio()
                logging.info('Step %d. Sample time: %s. Step time: %s. Replay ratio: %s.' % (
                             i, sample_time, step_time, replay_ratio))
                agent_summaries = self._agent.update_summaries()
                env_summaries = self._env_runner.summaries()
                self._writer.add_summaries(i, agent_summaries + env_summaries + buffer_summaries)

                for r_i, wrapped_buffer in enumerate(self._wrapped_buffer):
                    self._writer.add_scalar(
                        i, 'replay%d/add_count' % r_i,
                        wrapped_buffer.replay_buffer.add_count)
                    self._writer.add_scalar(
                        i, 'replay%d/size' % r_i,
                        wrapped_buffer.replay_buffer.replay_capacity
                        if wrapped_buffer.replay_buffer.is_full()
                        else wrapped_buffer.replay_buffer.add_count)

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

            self._writer.end_iteration()

            if i % self._save_freq == 0 and self._weightsdir is not None:
                self._save_model(i)

        if self._writer is not None:
            self._writer.close()

        logging.info('Stopping envs ...')
        self._env_runner.stop()
        [r.replay_buffer.shutdown() for r in self._wrapped_buffer]

    def validate_context(self, step):
        context_batch = next(self.ctxt_val_iter)
        val_info = self._agent.validate_context(step, context_batch)
        return 


