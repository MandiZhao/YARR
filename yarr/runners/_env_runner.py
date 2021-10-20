""" Note how episode rollout videos are logged/generated: custom_rlbench_env_multitask records episodes,
whenever terminal, returns as 'summaries' in act_result and send to multitask_rollout_generator, then 
stay as part of _EnvRunner.stored_transitions and get returned to stat_accumulator by EnvRunner  """
import copy
import logging
import os
import time
from multiprocessing import Process, Manager
from typing import Any, List, Union 

import numpy as np
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from multiprocessing import get_start_method, set_start_method


try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch 
WAIT_WARN=200

class _EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 online_task_ids, # limit the task options for env runner 
                 weightsdir: str = None,
                 device_list: List[int] = None, 
                 ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._agent = agent
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._episodes = episodes
        self._episode_length = episode_length
        self._rollout_generator = rollout_generator
        self._weightsdir = weightsdir
        self._previous_loaded_weight_folder = ''

        self._timesteps = timesteps

        self._p_args = {}
        self.p_failures = {}
        manager = Manager()
        self.write_lock = manager.Lock()
        self.stored_transitions = manager.list()
        self.agent_summaries = manager.list()
        self._kill_signal = kill_signal
        self._step_signal = step_signal
        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self.online_task_ids = online_task_ids 

        self._device_list, self._num_device = (None, 1) if device_list is None else (
            [torch.device("cuda:%d" % int(idx)) for idx in device_list], len(device_list))
        print('Internal EnvRunner is using GPUs:', self._device_list)
         

    def restart_process(self, name: str):
        p = Process(target=self._run_env, args=self._p_args[name], name=name)
        p.start()
        return p

    def spin_up_envs(self, name: str, num_envs: int, eval: bool):

        ps = []
        for i in range(num_envs):
            n = name + str(i)
            self._p_args[n] = (n, eval)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            ps.append(p)
        return ps
    
    def spinup_train_and_eval(self, n_train, n_eval, name='env'):
        ps = []
        i = 0
        # num_cpus = os.cpu_count()
        # print(f"Found {num_cpus} cpus, limiting:")
        # per_proc = int(num_cpus / (n_train+n_eval))
        for i in range(n_train):
            n = 'train_' + name + str(i)
            self._p_args[n] = (n, False, i)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start() 
            # print(os.system(f"taskset -cp {int(i * per_proc)}-{int( (i+1) * per_proc )} {p.pid}" ))
            ps.append(p)
        
        for j in range(n_train, n_train + n_eval):
            n = 'eval_' + name + str(j)
            self._p_args[n] = (n, True, j)
            self.p_failures[n] = 0
            p = Process(target=self._run_env, args=self._p_args[n], name=n)
            p.start()
            # print(os.system(f"taskset -cp {int(j * per_proc)}-{min(num_cpus-1, int( (j+1) * per_proc )) } {p.pid}" ))
            ps.append(p)
        return ps

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # Only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # Rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def _run_env(self, name: str, eval: bool, proc_idx: int):
        
        self._name = name

        self._agent = copy.deepcopy(self._agent)
        
        proc_device = self._device_list[int(proc_idx % self._num_device)] if self._device_list is not None else None
        self._agent.build(training=False, device=proc_device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._train_env
        if eval:
            env = self._eval_env
        env.eval = eval
        env.launch()
        for ep in range(self._episodes):
            self._load_save()
            logging.debug('%s: Starting episode %d.' % (name, ep))
            if not eval and len(self.online_task_ids) > 0:
                # print(f"env runner setting online tasks: {self.online_task_ids}")
                env.set_avaliable_tasks(self.online_task_ids) 
            episode_rollout = []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps, eval)
            try:
                for replay_transition in generator:
                    slept = 0
                    while True:
                        if self._kill_signal.value:
                            env.shutdown()
                            return 
                        if (eval or self._target_replay_ratio is None or
                                self._step_signal.value <= 0 or (
                                        self._current_replay_ratio.value >
                                        self._target_replay_ratio)):
                            break
                        time.sleep(1)
                        slept += 1
                        logging.info(
                            'Agent. Waiting for replay_ratio %f to be more than %f' %
                            (self._current_replay_ratio.value, self._target_replay_ratio))

                        if slept % WAIT_WARN == 0:
                            logging.warning(
                            'Env Runner process %s have been waiting for replay_ratio %f to be more than %f for %d seconds' %
                            (name, self._current_replay_ratio.value, self._target_replay_ratio, slept))
                            

                    with self.write_lock:
                        # logging.warning(f'proc {name}, idx {proc_idx} writing agent summaries')
                        if len(self.agent_summaries) == 0:
                            # Only store new summaries if the previous ones
                            # have been popped by the main env runner.
                            for s in self._agent.act_summaries():
                                self.agent_summaries.append(s)
                        # logging.warning(f'proc {name}, idx {proc_idx} finished writing agent summaries')
                    
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                env.shutdown()
                raise e

            with self.write_lock: 
                # logging.warning(f'proc {name}, idx {proc_idx} adding to stored transitions')
                for transition in episode_rollout:
                    self.stored_transitions.append((name, transition, eval))
                # logging.warning(f'proc {name}, idx {proc_idx} finished adding to stored transitions') 
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
