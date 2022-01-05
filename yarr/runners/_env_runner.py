""" Note how episode rollout videos are logged/generated: custom_rlbench_env_multitask records episodes,
whenever terminal, returns as 'summaries' in act_result and send to multitask_rollout_generator, then 
stay as part of _EnvRunner.stored_transitions and get returned to stat_accumulator by EnvRunner  """
import copy
import logging
import os
import time
from multiprocessing import Value, Process, Manager
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
CHECKPT='agent_checkpoint'

class _EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 episodes: int,
                 eval_episodes: int, 
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
                 all_task_var_ids = None,
                 ):
        self._train_env = train_env
        self._eval_env = eval_env
        self._agent = agent
        self._train_envs = train_envs
        self._eval_envs = eval_envs
        self._episodes = episodes
        self._eval_episodes = eval_episodes # evaluate each agent checkpoint this num eps for each task variation 
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

        self.stored_ckpt_eval_transitions = manager.dict() 
        self.agent_ckpt_eval_summaries = manager.dict()

        self._kill_signal = kill_signal
        self._step_signal = step_signal

        self._agent_checkpoint = Value('i', -1)
        self._loaded_eval_checkpoint = manager.list()
        self._finished_eval_checkpoint = manager.list()

        self._save_load_lock = save_load_lock
        self._current_replay_ratio = current_replay_ratio
        self._target_replay_ratio = target_replay_ratio
        self.online_task_ids = online_task_ids 
        self._all_task_var_ids = all_task_var_ids

        self._device_list, self._num_device = (None, 1) if device_list is None else (
            [torch.device("cuda:%d" % int(idx)) for idx in device_list], len(device_list))
        print('Internal EnvRunner is using GPUs:', self._device_list)
        self.online_buff_id = Value('i', -1)
         

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
    
    def spinup_train_and_eval(self, n_train, n_eval, name='env', iter_eval=False):
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
            p = Process(target=(self._iterate_all_vars if iter_eval else self._run_env), args=self._p_args[n], name=n)
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
                            logging.warning('_EnvRunner: agent hasnt finished writing.')
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _load_next_unevaled(self):
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
                    weight_folders = [w for w in weight_folders if w not in self._loaded_eval_checkpoint ]
                    # load the next unevaluated checkpoint 
                     
                if len(weight_folders) > 0:
                    w = weight_folders[0]  
                    self._agent_checkpoint.value = int(w)
                    d = os.path.join(self._weightsdir, str(w))
                    try:
                        self._agent.load_weights(d)
                    except FileNotFoundError:
                        # Rare case when agent hasn't finished writing.
                        logging.warning('_EnvRunner: agent hasnt finished writing.')
                        time.sleep(1)
                        self._agent.load_weights(d)
                    logging.info('Agent %s: Loaded weights: %s for evaluation' % (self._name, d)) 
                    with self.write_lock:
                        self._loaded_eval_checkpoint.append(w) 
                    break 
            logging.info('Waiting for weights to become available.') 
            if self._kill_signal.value:
                logging.info('Stop looking for new saved checkpoints before shutting down')
                return 
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
                logging.debug(f"env runner setting online tasks: {self.online_task_ids}")
                env.set_avaliable_tasks(self.online_task_ids) 
            if not eval and self.online_buff_id.value > -1:
                task_id, var_id = self._all_task_var_ids[self.online_buff_id.value] 
                env.set_task_variation(task_id, var_id)
            episode_rollout = []
            generator = self._rollout_generator.generator(
                self._step_signal, env, self._agent,
                self._episode_length, self._timesteps, eval, 
                swap_task=(False if not eval and self.online_buff_id.value > -1 else True)
                )
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

    def _iterate_all_vars(self, name: str,  eval: bool, proc_idx: int): 
        # use for eval env only 
        self._name = name
        self._agent = copy.deepcopy(self._agent)
        proc_device = self._device_list[int(proc_idx % self._num_device)] if self._device_list is not None else None
        self._agent.build(training=False, device=proc_device)
        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)
        ckpt = None 
        env = self._eval_env
        env.eval = True 
        env.launch()
         
        while True:
            if self._kill_signal.value:
                logging.info('shutting down before loading new ckpt')                
                env.shutdown()
                return 
            self._load_next_unevaled()
            if self._kill_signal.value and ckpt is not None and ckpt == int(self._agent.get_checkpoint()) :
                logging.info('No new checkpoint got loaded, shutting down')  
                env.shutdown()
                return 
            ckpt = int(self._agent.get_checkpoint()) 
            assert ckpt not in self.stored_ckpt_eval_transitions.keys(), 'There should be no transitions stored for this ckpt'
            # with self.write_lock:
            #     self.stored_ckpt_eval_transitions[ckpt] = []
            #     self.agent_ckpt_eval_summaries[ckpt] = [] 
            all_episode_rollout = []
            all_agent_summaries = []
            for (task_id, var_id) in self._all_task_var_ids:
                if self._kill_signal.value: 
                    logging.info('[Finishing evaluation before full shutdown] process', name, 'evaluating task + var:', task_id, var_id)
                env.set_task_variation(task_id, var_id)
                for ep in range(self._eval_episodes):
                    # print('%s: Starting episode %d.' % (name, ep))
                    episode_rollout = []
                    generator = self._rollout_generator.generator(
                        self._step_signal, env, self._agent,
                        self._episode_length, self._timesteps, True, 
                        swap_task=False)
                    try:
                        for replay_transition in generator:    
                            for s in self._agent.act_summaries():
                                s.step = ckpt
                                all_agent_summaries.append(s)
                                # logging.warning(f'proc {name}, idx {proc_idx} finished writing agent summaries')
                            assert replay_transition.info[CHECKPT] == ckpt, 'Checkpoint mismatch between transition in rollout and agent loaded point'
                            episode_rollout.append(replay_transition)
                            # print(replay_transition.info, env._task._variation_number )
                    except StopIteration as e:
                        continue
                    except Exception as e:
                        env.shutdown()
                        raise e

                    for transition in episode_rollout:
                        all_episode_rollout.append((name, transition)) 

            with self.write_lock: 
                self.stored_ckpt_eval_transitions[ckpt] = all_episode_rollout  
                self.agent_ckpt_eval_summaries[ckpt] = all_agent_summaries
                self._finished_eval_checkpoint.append(ckpt)

            if self._kill_signal.value:
                print('shutting down after current ckpt is done evaluating')
                env.shutdown()
                return 
             
            print(f'Checkpoint {ckpt} finished evaluating, all {len(self.stored_ckpt_eval_transitions[ckpt])} transitions and agent act summaries stored ') 
        env.shutdown()

        


    def kill(self):
        self._kill_signal.value = True
