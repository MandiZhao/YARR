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
DEMO_KEY='front_rgb'
TASK_ID='task_id'
VAR_ID='variation_id'
CHECKPT='agent_checkpoint'
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

NOISY_VECS_20={
    0: np.array([ 0.1241162 , -0.05574423, -0.05697403, -0.17554671,  0.19445307,
       -0.35501783,  0.20034263,  0.01890455, -0.23385913,  0.04746695,
       -0.32926621,  0.21168267,  0.26189741, -0.16748583, -0.2208488 ,
       -0.342916  , -0.34976264, -0.15677703, -0.1557932 , -0.31419748]),
    1: np.array([ 0.22598381,  0.21672723, -0.26101817,  0.02933299,  0.11744386,
        0.23375912,  0.05646577,  0.34363911,  0.0057265 ,  0.25028927,
        0.30709834, -0.3051258 , -0.08236424,  0.03895639, -0.360537  ,
        0.16915585, -0.028283  ,  0.36571162,  0.26494207,  0.14672129]),
    2: np.array([-0.37600589,  0.2670904 ,  0.10579851,  0.23620909, -0.16698364,
       -0.29006283,  0.11340786,  0.04280838, -0.30325462, -0.0377304 ,
       -0.30273325, -0.32708192,  0.05640233,  0.29489397,  0.19917879,
       -0.00207009, -0.24073129,  0.33234255,  0.05778171, -0.00117703]),
    3: np.array([ 0.03727079,  0.29780001,  0.20769243,  0.24645232, -0.28714051,
       -0.01010143, -0.01264227,  0.36535678, -0.2685693 ,  0.35804983,
        0.38971603, -0.03167043,  0.11421525,  0.13804317, -0.13793809,
        0.05940818, -0.10723408, -0.23744542,  0.28166427, -0.18647186]),
    4: np.array([ 0.27144968,  0.15943166,  0.27796568, -0.05611184,  0.18805349,
       -0.33940988,  0.28816161, -0.17185427,  0.35198909, -0.15380081,
        0.19273471, -0.15989264,  0.14688273,  0.06881908,  0.34219501,
       -0.05920143,  0.22709361,  0.27779321,  0.06714273, -0.2586969 ]),
    5: np.array([ 0.34249773,  0.06390457, -0.13743363,  0.38286517,  0.1551175 ,
       -0.18142844,  0.16433129, -0.18968841, -0.06506652, -0.20942131,
        0.36155665,  0.11065518, -0.32124776,  0.22070202, -0.09678616,
        0.38956303, -0.16748455,  0.10717698,  0.21597242, -0.05381803]),
    6: np.array([ 0.23345451, -0.11720424,  0.27668721,  0.30169692, -0.25731543,
       -0.33149893,  0.30340655,  0.25515817, -0.25552404,  0.04616548,
        0.1204444 ,  0.12879893, -0.31810688,  0.13517071, -0.05832574,
       -0.11645197, -0.30803733, -0.22775973, -0.14239043,  0.17011443]),
    7: np.array([-0.18991155,  0.35840933, -0.33248923,  0.24374355,  0.30816331,
       -0.3521371 , -0.24067894, -0.22872415,  0.12738857,  0.01691464,
       -0.131966  ,  0.18401505, -0.17695116, -0.06359497,  0.03967199,
       -0.14142925,  0.26656281,  0.20365368,  0.07496383,  0.30567518]),
    8: np.array([ 0.11485718, -0.31645387,  0.31761709, -0.35623829, -0.00555676,
       -0.08726689,  0.01714125, -0.06494054, -0.26445658,  0.18859192,
       -0.01895199, -0.3232076 ,  0.12715038, -0.17051961,  0.37937627,
        0.26442016,  0.14463371,  0.17563099, -0.2184145 ,  0.27899077]),
    9: np.array([-0.05557781, -0.34685366,  0.33585084,  0.11439378, -0.27021591,
        0.05677191, -0.30173682,  0.34768466,  0.04299883, -0.19551378,
        0.04949826,  0.0218476 , -0.0590866 ,  0.14459832, -0.26284077,
        0.35558838, -0.3428256 , -0.02309729, -0.26602248, -0.10388196])
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
        num_task_vars=20,
        task_var_to_replay_idx=dict(),
        dev_cfgs={}, 
        augment_reward=False, 
        ):
        self._demo_dataset = demo_dataset 
        if demo_dataset is None:
            print('Warning! RolloutGenerator not sampling context from demo dataset')
        self._sample_key = sample_key 
        self._one_hot = one_hot 
        self._noisy_one_hot = noisy_one_hot 
        self._num_task_vars = num_task_vars 
        self._task_var_to_replay_idx = task_var_to_replay_idx
        self._dev_cfgs = dev_cfgs
        self._augment_reward = augment_reward

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
 

    def sample_context(self, task_id, variation_id, task_name):
        """takes in an task_id from environment and sample some 
            demos from offline dataset"""
        assert self._demo_dataset is not None, 'Cannot sample without demo dataset pre-loaded'
        # data = self._demo_dataset.sample_one_variation(task_id, variation_id)[0]
        # NOTE: change to sample multiple here!
        demo_sample = self._demo_dataset.sample_for_replay_no_match(task_id, variation_id) # draw K independent samples 
        demo_sample = torch.stack( [ d[DEMO_KEY] for d in demo_sample ], dim=0)
        # assert task_name in data['name'], f"Expects {task_name} to be the prefix of {data['name']}"
        # demo_sample = data.get(self._sample_key, None)
        # assert demo_sample is not None, f"Key {self._sample_key} was not found in sampled data"
        return demo_sample 

    
    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool, swap_task: bool = True): 
        obs     = env.reset(swap_task=swap_task) 
        task_id = env._active_task_id
        variation_id = env._active_variation_id
        task_name = env._active_task_name
        buf_id = self._task_var_to_replay_idx[task_id][variation_id]
        #print('mt rollout gen:', task_id, variation_id, task_name)
        agent.reset()
        checkpoint = agent.get_checkpoint()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        demo_samples = None
        one_hot_vec = F.one_hot(  torch.tensor(int(buf_id)), num_classes=self._num_task_vars)
        if self._noisy_one_hot:
            assert int(variation_id) in NOISY_VECS.keys(), 'Support only 10 variations for now!!'
            noisy_one_hot = torch.tensor(NOISY_VECS[int(variation_id)]).clone().detach().to(torch.float32)
            if self._dev_cfgs.get('noisy_dim_20', False):
                noisy_one_hot = torch.tensor(NOISY_VECS_20[int(variation_id)]).clone().detach().to(torch.float32)
        one_hot_vec = one_hot_vec.clone().detach().to(torch.float32)
        episode_trans = [] 
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
            if self._noisy_one_hot or self._dev_cfgs.get('noisy_dim_20', False):
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
                CHECKPT: checkpoint
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
            episode_trans.append(replay_transition)
            # yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                # return
                break 

        if self._augment_reward and episode_trans[-1].reward > 0:
            eps_len = len(episode_trans)
            # print(f'Generated successfull traj of length {eps_len}, relabeling the reward')
            for i, ep in enumerate(episode_trans):
                ep.reward = max(i/eps_len, 0.1) * episode_trans[-1].reward

        for ep in episode_trans: 
            yield ep 