from multiprocessing import Value

import numpy as np

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.envs.env import MultiTaskEnv
from yarr.envs.rlbench_env import MultiTaskRLBenchEnv
from yarr.utils.transition import ReplayTransition
import torch 
import torch.nn.functional as F
CONTEXT_KEY = 'demo_sample' 
TASK_ID='task_id'
VAR_ID='variation_id'
NOISY_VECS={
    0: np.array([ 0.11925248, -0.14782887, -0.40366346, -0.27328304, -0.42401568,
        0.33704594,  0.41446234,  0.48869492,  0.10952299,  0.10154486]),
    1: np.array([ 0.22487479,  0.32561551, -0.22771661, -0.04798632, -0.43327144,
        0.35466405,  0.03087024,  0.00246092, -0.3818759 , -0.57354108]),
    2: np.array([-0.38734519, -0.31740813, -0.36871456, -0.14102713, -0.17094035,
        0.32675689, -0.21493319,  0.28633581,  0.24948758, -0.51667931]),
    3: np.array([ 0.18540121, -0.36820533,  0.18308575, -0.01892064, -0.33780466,
        0.26526095,  0.53540062,  0.2748794 ,  0.24834015,  0.43337298]),
    4: np.array([ 0.2148131 , -0.25162006, -0.30730072,  0.50525774,  0.45081893,
       -0.49889678, -0.21654777, -0.16233489,  0.11915335, -0.03528155]), 
    5: np.array([-0.40118621,  0.41678064,  0.28864454, -0.53277101, -0.12082425,
        0.4367015 , -0.27647954, -0.05299221,  0.0547752 ,  0.10308621]),
    6: np.array([-0.02411682, -0.11224046,  0.03227223,  0.11851931, -0.40225017,
        0.46261037, -0.428421  , -0.47695132, -0.20697166, -0.37690079]),
    7: np.array([-0.32398346,  0.18281037,  0.08709628, -0.44767024,  0.40639542,
        0.23711536,  0.40272094,  0.17245336, -0.18032934,  0.45584731]),
    8: np.array([ 0.25014659,  0.25815153, -0.09170533, -0.33635687,  0.50909926,
        0.57739529,  0.22484016, -0.1269459 , -0.11276835,  0.278004  ]),
    9: np.array([-0.18245398, -0.22966254,  0.4778035 , -0.38738104, -0.39714424,
        0.32174066,  0.17275781,  0.32849225, -0.13299022, -0.34485649])
}
class RolloutGeneratorWithContext(object):
    """For each env step, also sample from the demo dataset to 
        generate context embeddings"""

    def __init__(
        self, 
        demo_dataset=None, 
        sample_key='front_rgb', 
        one_hot=False, 
        noisy_one_hot=False,
        num_vars=20):
        self._demo_dataset = demo_dataset 
        if demo_dataset is None:
            print('Warning! RolloutGenerator not sampling context from demo dataset')
        self._sample_key = sample_key 
        self._one_hot = one_hot 
        self._noisy_one_hot = noisy_one_hot 
        self._num_vars = num_vars 

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
 

    def sample_context(self, task_id, variation_id, task_name):
        """takes in an task_id from environment and sample some 
            demos from offline dataset"""
        assert self._demo_dataset is not None, 'Cannot sample without demo dataset pre-loaded'
        data = self._demo_dataset.sample_one_variation(task_id, variation_id)[0]
        # data = self._demo_dataset.sample_one_variation(
        #     task_id, variation_id, size=5)  # NOTE(sample multiple here!)
        assert task_name in data['name'], f"Expects {task_name} to be the prefix of {data['name']}"
        demo_sample = data.get(self._sample_key, None)
        assert demo_sample is not None, f"Key {self._sample_key} was not found in sampled data"
        return demo_sample

    
    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool): 
        obs     = env.reset()
        task_id = env._active_task_id
        variation_id = env._active_variation_id
        task_name = env._active_task_name
        #print('mt rollout gen:', task_id, variation_id, task_name)
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        demo_samples = None
        one_hot_vec = F.one_hot(  torch.tensor(int(variation_id)), num_classes=self._num_vars)
        if self._noisy_one_hot:
            assert int(variation_id) in NOISY_VECS.keys(), 'Support only 10 variations for now!!'
            noisy_one_hot = torch.tensor(NOISY_VECS[int(variation_id)]).clone().detach().to(torch.float32)
        one_hot_vec = one_hot_vec.clone().detach().to(torch.float32)
        for step in range(episode_length):

            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}
            prepped_data.update({
                TASK_ID: task_id, 
                VAR_ID: variation_id})
            
            if self._demo_dataset is not None:
                demo_samples = self.sample_context(task_id, variation_id, task_name)
                prepped_data[CONTEXT_KEY] = demo_samples

            if self._one_hot:
                prepped_data[CONTEXT_KEY] = one_hot_vec   
            if self._noisy_one_hot:
                prepped_data[CONTEXT_KEY] = noisy_one_hot
            # print('rollout generator input:', prepped_data.keys())
            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_extra_elems = {k: np.array(v) for k, v in
                                 act_result.replay_elements.items()}

            transition = env.step(act_result)
            assert env._active_task_name == task_name and env._active_task_id == task_id and env._active_variation_id == variation_id, \
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
            # obs.update(agent_extra_elems)
            obs_tp1 = dict(transition.observation)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info.update({
                TASK_ID: task_id, 
                VAR_ID: variation_id, #'task_name': env._active_task_name
                'demo': False,
                })

            replay_transition = ReplayTransition(
                observation=obs, action=act_result.action, reward=transition.reward,
                terminal=transition.terminal, timeout=timeout,  
                info=transition.info,
                summaries=transition.summaries,)

            # if transition.terminal:
            #     print('rollout gen got transition:', transition.summaries)
                  
            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: np.array([v]) for k, v in
                                    obs_history.items()}
                    if self._demo_dataset is not None:
                        prepped_data[CONTEXT_KEY] = demo_samples

                    if self._one_hot:
                        prepped_data[CONTEXT_KEY] = one_hot_vec
                    if self._noisy_one_hot:
                        prepped_data[CONTEXT_KEY] = noisy_one_hot

                    prepped_data.update({
                        TASK_ID: task_id, 
                        VAR_ID: variation_id})
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
