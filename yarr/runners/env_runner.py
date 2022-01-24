import collections
import logging
import os
import signal
import time
from multiprocessing import Value, Manager
from threading import Thread
from typing import List
from typing import Union

import numpy as np

from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary, ImageSummary, VideoSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners._env_runner import _EnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator

import torch 
from copy import deepcopy 
TASK_ID='task_id'
VAR_ID='variation_id'
WAIT_TIME=2000 # original version was 600 -> 5min

class EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 agent: Agent,
                 train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
                 num_train_envs: int,
                 num_eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 eval_env: Union[Env, None] = None,
                 eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 max_fails: int = 5,
                 device_list: Union[List[int], None] = None,
                 share_buffer_across_tasks: bool = True, 
                 task_var_to_replay_idx: dict = {},
                 eval_only: bool = False, 
                 iter_eval: bool = False, 
                 eval_episodes: int = 2,
                 log_freq: int = 100,
                 target_replay_ratio: float = 30.0,
                 final_checkpoint_step: int = 999,
                 dev_cfg: dict = None,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env if eval_env else deepcopy(train_env)
        self._agent = agent
        self._train_envs = num_train_envs
        self._eval_envs = num_eval_envs
        self._train_replay_buffer = train_replay_buffer if isinstance(train_replay_buffer, list) else [train_replay_buffer]
        self._timesteps = self._train_replay_buffer[0].timesteps
        if eval_replay_buffer is not None:
            eval_replay_buffer = eval_replay_buffer if isinstance(eval_replay_buffer, list) else [eval_replay_buffer]
        self._eval_replay_buffer = eval_replay_buffer
        self._episodes = episodes
        self._episode_length = episode_length
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (
            RolloutGenerator() if rollout_generator is None
            else rollout_generator)
        self._weightsdir = weightsdir
        self._max_fails = max_fails
        self._previous_loaded_weight_folder = ''
        self._p = None
        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_episodes = {'train_envs': 0, 'eval_envs': 0} 
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1) 
        self.online_task_ids = Manager().list()
        self.buffer_add_counts = Manager().list()
        self.log_freq = log_freq
        self.target_replay_ratio = target_replay_ratio
        self._device_list = device_list 
        self._share_buffer_across_tasks = share_buffer_across_tasks 
        self._agent_summaries = []
        self._agent_ckpt_summaries = dict() 
        self.task_var_to_replay_idx = task_var_to_replay_idx
        self._all_task_var_ids = []
        for task_id, var_dicts in task_var_to_replay_idx.items():
            self._all_task_var_ids.extend([(task_id, var_id) for var_id in var_dicts.keys() ])
        logging.info(f'Counted a total of {len(self._all_task_var_ids)} variations')
        
        self._eval_only = eval_only
        if eval_only:
            logging.info('Warning! Eval only, set number of training env to 0')
            self._train_envs = 0

        self._iter_eval = iter_eval
        self._eval_episodes = eval_episodes
        self.final_checkpoint_step = final_checkpoint_step
        self._dev_cfg = dev_cfg

    @property   
    def device_list(self):
        # if self._device_list is None:
        #     return [i for i in range(torch.cuda.device_count())]
        # NOTE: if never given gpus at __init__, don't use gpus even if some are avaliable for agent training 
        return deepcopy(self._device_list)
    
    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(
                ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(
                ScalarSummary('%s/total_transitions' % key, value))
        for key, value in self._total_episodes.items():
            summaries.append(
                ScalarSummary('%s/total_episodes' % key, value))
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)
        
        return summaries

    def _update(self):
        # Move the stored transitions to the replay and accumulate statistics.
        new_transitions = collections.defaultdict(int)
        
        with self._internal_env_runner.write_lock:
            # logging.info('EnvRunner calling internal runner write lock')
            self._agent_summaries = list(
                self._internal_env_runner.agent_summaries)
            if self._step_signal.value % self.log_freq == 0 and self._step_signal.value > 0:
                self._internal_env_runner.agent_summaries[:] = []
            
            for _id, buffer in enumerate(self._train_replay_buffer):
                context_inputs = {}
                bsize, wsize = self._dev_cfg.pearl_context_size, self._dev_cfg.pearl_window_size
                if buffer.context_avaliable(bsize, wsize):
                    recent_batch = buffer.sample_recent_batch(bsize, wsize) 
                    context_inputs = {
                    'context_action': torch.tensor(recent_batch['action']),
                    'context_reward': torch.tensor(recent_batch['reward']).unsqueeze(1)
                    }
                    # recent_batcqh['front_rgb'].shape (B, 1, 3, 128, 128)
                    obs = torch.cat(
                            [torch.tensor(recent_batch['front_rgb']), 
                            torch.tensor(recent_batch['front_rgb_tp1'])], dim=1)
                    context_inputs['context_obs'] = (obs.float() / 255.0) * 2.0 - 1.0
                self._internal_env_runner._context_batches[_id] = context_inputs
            
            for name, transition, eval in self._internal_env_runner.stored_transitions:
                add_to_buffer = (not eval) or self._eval_replay_buffer is not None
                if self._train_envs == 0 and self._iter_eval:
                    add_to_buffer = True # for PEARL agent, need to update buffer with eval transitions!
                if add_to_buffer:
                    kwargs = dict(transition.observation)
                    kwargs.update(transition.info)
                    # assert self._buffer_key in transition.info.keys(), \
                    #     f'Need to look for **{self._buffer_key}** in replay transition to know which buffer to add it to'
                    # replay_index = self.task_var_to_replay_idx.get(
                    #     transition.info[self._buffer_key], 0)
                    task_id = transition.info.get(TASK_ID, 0)
                    var_id = transition.info.get(VAR_ID, 0) 
                    replay_index = self.task_var_to_replay_idx[task_id][var_id]
                    if self._share_buffer_across_tasks:
                        replay_index = 0
                    rb = self._train_replay_buffer[replay_index]
                    rb.add(
                        np.array(transition.action), transition.reward,
                        transition.terminal,
                        transition.timeout, **kwargs)
                    if transition.terminal:
                        rb.add_final(
                            **transition.final_observation)
                new_transitions[name] += 1
                self._new_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                self._total_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                
                if transition.terminal:
                    self._total_episodes['eval_envs' if eval else 'train_envs'] += 1

                if self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
            self._internal_env_runner.stored_transitions[:] = []  # Clear list
            # logging.info('Finished EnvRunner calling internal runner write lock')
            
            for ckpt_step, all_transitions in self._internal_env_runner.stored_ckpt_eval_transitions.items():
                if ckpt_step in self._internal_env_runner._finished_eval_checkpoint:  
                    for name, transition in all_transitions:
                        self._new_transitions['eval_envs'] += 1
                        self._total_transitions['eval_envs'] += 1
                        if transition.terminal:
                            self._total_episodes['eval_envs'] += 1 
                    
                    self._agent_ckpt_summaries[ckpt_step] = self._internal_env_runner.agent_ckpt_eval_summaries.pop(ckpt_step, [])
                    if self._stat_accumulator is not None:
                        self._stat_accumulator.step_all_transitions_from_ckpt(all_transitions, ckpt_step)
                    self._internal_env_runner.stored_ckpt_eval_transitions.pop(ckpt_step, []) # Clear 
                    

                    logging.debug('Done poping ckpt {} eval transitions to accumulator, main EnvRunner stored {} agent summaries, remaining ckpts: '.format(
                        ckpt_step, 
                        len(self._agent_ckpt_summaries.get(ckpt_step, []))), 
                        self._internal_env_runner.stored_ckpt_eval_transitions.keys() 
                        )

            self.buffer_add_counts[:] = [int(r.add_count) for r in self._train_replay_buffer]
            demo_cursor = self._train_replay_buffer[0]._demo_cursor
            if demo_cursor > 0: # i.e. only on-line samples can be used for context
                self.buffer_add_counts[:] = [int(r.add_count - r._demo_cursor) for r in self._train_replay_buffer]
            self._internal_env_runner.online_buff_id.value = -1 
            if self._train_replay_buffer[0].batch_size > min(self.buffer_add_counts):
                buff_id = np.argmin(self.buffer_add_counts) 
                self._internal_env_runner.online_buff_id.value = buff_id
             
        return new_transitions
 
    def try_log_ckpt_eval(self):
        """Attempts to log the earliest avaliable ckpt that finished eval"""
        ckpt, summs = self._stat_accumulator.pop_ckpt_eval() 
        if ckpt > -1:
            assert ckpt in self._agent_ckpt_summaries.keys(), 'Checkpoint has env transitions all stepped in accumulator but no agent summaries found' 
            summs += self._agent_ckpt_summaries.pop(ckpt, [])
        return ckpt, summs 

    def _run(self, save_load_lock):
        self._internal_env_runner = _EnvRunner(
            train_env=self._train_env, eval_env=self._eval_env, agent=self._agent, timesteps=self._timesteps, train_envs=self._train_envs,
            eval_envs=self._eval_envs, episodes=self._episodes, episode_length=self._episode_length, kill_signal=self._kill_signal,
            step_signal=self._step_signal, rollout_generator=self._rollout_generator, save_load_lock=save_load_lock,
            current_replay_ratio=self.current_replay_ratio, 
            target_replay_ratio=self.target_replay_ratio, 
            online_task_ids=self.online_task_ids,
            weightsdir=self._weightsdir, 
            device_list=(self.device_list if len(self.device_list) >= 1 else None),
            all_task_var_ids=self._all_task_var_ids,
            eval_episodes=self._eval_episodes,
            final_checkpoint_step=self.final_checkpoint_step, 
            )
        #training_envs = self._internal_env_runner.spin_up_envs('train_env', self._train_envs, False)
        #eval_envs = self._internal_env_runner.spin_up_envs('eval_env', self._eval_envs, True)
        #envs = training_envs + eval_envs
        envs = self._internal_env_runner.spinup_train_and_eval(self._train_envs, self._eval_envs, 'env', iter_eval=self._iter_eval)
        no_transitions = {env.name: 0 for env in envs}
        while True:
            for p in envs:
                if p.exitcode is not None:
                    envs.remove(p)
                    if p.exitcode != 0:
                        self._internal_env_runner.p_failures[p.name] += 1
                        n_failures = self._internal_env_runner.p_failures[p.name]
                        if n_failures > self._max_fails:
                            logging.error('Env %s failed too many times (%d times > %d)' %
                                          (p.name, n_failures, self._max_fails))
                            raise RuntimeError('Too many process failures.')
                        logging.warning('Env %s failed (%d times <= %d). restarting' %
                                        (p.name, n_failures, self._max_fails))
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)

            if not self._kill_signal.value or len(self._internal_env_runner.stored_transitions) > 0 or \
                 len(self._internal_env_runner.stored_ckpt_eval_transitions) > 0:
                new_transitions = self._update()
                for p in envs:
                    if new_transitions[p.name] == 0:
                        no_transitions[p.name] += 1
                    else:
                        no_transitions[p.name] = 0
                    if no_transitions[p.name] > WAIT_TIME:  # 5min
                        if self.current_replay_ratio.value - 1 > self.target_replay_ratio:
                            # only hangs if it Should be running, otherwise just let it sleep? 
                            logging.warning("Env %s hangs, so restarting" % p.name)
                            print('process id:', p.pid)
                            print('process is alive?', p.is_alive())
                            print('replay&target ratios:', self.current_replay_ratio.value, self.target_replay_ratio)
                            envs.remove(p)
                            os.kill(p.pid, signal.SIGTERM)
                            torch.cuda.empty_cache()
                            p = self._internal_env_runner.restart_process(p.name)
                            envs.append(p)
                            no_transitions[p.name] = 0

            if len(envs) == 0:
                break
            time.sleep(1)

    def start(self, save_load_lock):
        self._p = Thread(target=self._run, args=(save_load_lock,), daemon=True)
        self._p.name = 'EnvRunnerThread'  
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()

    def set_step(self, step):
        self._step_signal.value = step
