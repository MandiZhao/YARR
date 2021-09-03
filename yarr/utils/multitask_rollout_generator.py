from multiprocessing import Value

import numpy as np

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.envs.env import MultiTaskEnv
from yarr.envs.rlbench_env import MultiTaskRLBenchEnv
from yarr.utils.transition import ReplayTransition
CONTEXT_KEY = 'demo_sample' 
TASK_ID='task_id'
VAR_ID='variation_id'

class RolloutGeneratorWithContext(object):
    """For each env step, also sample from the demo dataset to 
        generate context embeddings"""

    def __init__(self, demo_dataset, sample_key='front_rgb'):
        self._demo_dataset = demo_dataset 
        self._sample_key = sample_key 

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
 

    def sample_context(self, task_id, variation_id, task_name):
        """takes in an task_id from environment and sample some 
            demos from offline dataset"""
        data = self._demo_dataset.sample_one_variation(task_id, variation_id)
        assert task_name in data['name'], f"Expects {task_name} to be the prefix of {data['name']}"
        demo_sample = data.get(self.sample_key, None)
        assert demo_sample is not None, f"Key {self.sample_key} was not found in sampled data"
        return demo_sample

    
    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):
        
        obs     = env.reset()
        task_id = env._active_task_id
        variation_id = env._active_variation_id
        task_name = env._active_task_name
        print(task_id, variation_id, task_name)
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):

            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}
            prepped_data[CONTEXT_KEY] = self.sample_context(task_id, variation_id, task_name)
            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_extra_elems = {k: np.array(v) for k, v in
                                 act_result.replay_elements.items()}

            transition = env.step(act_result)
            assert env._active_task_name == task_name and env._active_task_id == task_id, \
                 'Something is wrong with RLBench Env, task {task_name} is replaced by {env.active_task_name} in middle of an episode'
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs.update(agent_obs_elems)
            obs_tp1 = dict(transition.observation)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info.update({
                TASK_ID: env._active_task_id, 
                VAR_ID: env._active_variation_id, #'task_name': env._active_task_name
                })

            replay_transition = ReplayTransition(
                obs, act_result.action, transition.reward,
                transition.terminal,
                timeout, obs_tp1, agent_extra_elems,
                transition.info)
 
            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: np.array([v]) for k, v in
                                    obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
