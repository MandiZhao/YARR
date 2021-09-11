from multiprocessing import Lock
from typing import List

import numpy as np
from yarr.agents.agent import Summary, ScalarSummary
from yarr.utils.transition import ReplayTransition

from collections import deque 
TASK_ID = 'task_id'
VAR_ID = 'variation_id'

class StatAccumulator(object):

    def step(self, transition: ReplayTransition, eval: bool):
        pass

    def pop(self) -> List[Summary]:
        pass

    def peak(self) -> List[Summary]:
        pass

    def reset(self) -> None:
        pass

class Metric(object):

    def __init__(self):
        self._previous = []
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        # self._previous.clear()
        return 

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]

class DequeMetric(Metric):
    """ Take mean over a fixed lengthh list of logged values """
    def __init__(self, length=5):
        self._previous = deque([], maxlen=length)
        self._current = 0 
        self._maxlen = length 
 

class _SimpleAccumulator(StatAccumulator):

    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._episode_returns = Metric()
        self._episode_lengths = Metric()
        self._summaries = []
        self._transitions = 0

    def _reset_data(self):
        with self._lock:
            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1
            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:
                self._episode_returns.next()
                self._episode_lengths.next()
            self._summaries.extend(list(transition.summaries))

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        names = ["return", "length"]
        metrics = [self._episode_returns, self._episode_lengths]
        for name, metric in zip(names, metrics):
            for stat_key in stat_keys:
                if self._mean_only:
                    assert stat_key == "mean"
                    sum_name = '%s/%s' % (self._prefix, name)
                else:
                    sum_name = '%s/%s/%s' % (self._prefix, name, stat_key)
                sums.append(
                    ScalarSummary(sum_name, getattr(metric, stat_key)()))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions))
        sums.extend(self._summaries)
        return sums

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 1:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()
    
    def reset(self):
        self._transitions = 0
        self._reset_data()


class SimpleAccumulator(StatAccumulator):

    def __init__(self, eval_video_fps: int = 30, mean_only: bool = True):
        self._train_acc = _SimpleAccumulator(
            'train_envs', eval_video_fps, mean_only=mean_only)
        self._eval_acc = _SimpleAccumulator(
            'eval_envs', eval_video_fps, mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition, eval)
        else:
            self._train_acc.step(transition, eval)

    def pop(self) -> List[Summary]:
        return self._train_acc.pop() + self._eval_acc.pop()

    def peak(self) -> List[Summary]:
        return self._train_acc.peak() + self._eval_acc.peak()
    
    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()


class SimpleMultiVariationAccumulator(StatAccumulator):
    """ Use for one task, evenly tracks its variations """
    def __init__(self, 
                 prefix: str, 
                 task_name: str,        # e.g. reach_target or push_button
                 task_id: int,          # e.g. 0 or 1 
                 task_vars: List[str],  # e.g. [reach_target_0, reach_target_1, ...]
                 eval_video_fps: int = 30,
                 mean_only: bool = True, 
                 max_len: int = 5,
                 task_id_key: str = 'task_id',
                 var_id_key: str = 'variation_id'):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._task_name = task_name
        self._task_id = task_id
        self._task_vars = task_vars 
        self._all_var_returns = {_var: DequeMetric(max_len) for _var in task_vars}
        self._all_var_lengths = {_var: DequeMetric(max_len) for _var in task_vars}
        self._summaries = []
        self._transitions = {_var: 0 for _var in task_vars}
        self._task_id_key, self._var_id_key = task_id_key, var_id_key

    def _reset_data(self):
        with self._lock:
            [metric.reset() for metric in self._all_var_returns]
            [metric.reset() for metric in self._all_var_lengths]
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        assert self._task_id_key in transition.info.keys() and \
            self._var_id_key in transition.info.keys(), 'Not seeing the data for task or variation ID'
    
        if transition.info[self._task_id_key] == self._task_id:
            var_str = f'{self._task_name}_{transition.info[self._var_id_key]}'
            # print('\n', var_str) 
            with self._lock:
                self._transitions[var_str] += 1
                self._all_var_returns[var_str].update(transition.reward)
                self._all_var_lengths[var_str].update(1)
                if transition.terminal:
                    self._all_var_returns[var_str].next()
                    self._all_var_lengths[var_str].next()
                self._summaries.extend(list(transition.summaries))

    def _get(self, _var) -> List[Summary]:
        sums = []
        stat_keys = ["mean"] if self._mean_only else \
            ["min", "max", "mean", "median", "std"]
        returns, lengths = self._all_var_returns[_var], self._all_var_lengths[_var]
    
        returns, lengths = self._all_var_returns[_var], self._all_var_lengths[_var]
        for stat_key in stat_keys:
            sum_name = f"{self._prefix}/{self._task_name}/{_var}/return/{stat_key}"
            sums.append(
                ScalarSummary(sum_name, getattr(returns, stat_key)()))

            sum_name = f"{self._prefix}/{self._task_name}/{_var}/length/{stat_key}"
            sums.append(
                ScalarSummary(sum_name, getattr(lengths, stat_key)())
                )
    
        sums.append(ScalarSummary(
            f"{self._prefix}/{self._task_name}/{_var}/num_transitions", self._transitions[_var] ))
         
        return sums

    # def pop(self) -> List[Summary]:
    #     data = []
    #     for k, v in self._all_var_returns.items():
    #         if len(v) == v._maxlen :
    #             data.extend(self._get(k))
    #             # self._reset_data() NOTE(mandi): shouldn't need to bc of deque
  
    #     data.append(
    #         ScalarSummary(f"{self._prefix}/{self._task_name}/total_transitions", \
    #             sum(list(self._transitions.values()) ) ) )
    #     data.extend(self._summaries)
    #     self._summaries.clear()
    #     return data
    def pop(self) -> List[Summary]:
        data = []
        for k, v in self._all_var_returns.items():
            if len(v) != v._maxlen :
                return []  # doesn't log if any variation doesn't have enough value saved
        
        all_var_ret, all_var_len, all_var_trans = [], [], []
        for _var in self._task_vars:
            returns, lengths = self._all_var_returns[_var], self._all_var_lengths[_var]
            all_var_ret.append( returns.mean() )
            all_var_len.append( lengths.mean() )
            all_var_trans.append(self._transitions[_var])
        
        for metric, values in zip(
            ["num_transitions", "return", "length"], [all_var_trans, all_var_ret, all_var_len]):
            data.append(
                ScalarSummary(f"{self._prefix}_envs/{self._task_name}/{metric}_mean", \
                np.mean(values))
                )

            data.append( 
                ScalarSummary(f"{self._prefix}_envs/{self._task_name}/{metric}_std", \
                np.std(values))
                )
        
        data.extend(self._summaries)
        self._summaries.clear()
        return data

    def peak(self) -> List[Summary]:
        # data = []
        # for k, v in self._all_var_returns.items():
        #     if len(v) == v._maxlen :
        #         data.extend(self._get(k))
        #         # self._reset_data() NOTE(mandi): shouldn't need to bc of deque
  
        # data.append(
        #     ScalarSummary(f"{self._prefix}/{self._task_name}/total_transitions", \
        #         sum(list(self._transitions.values()) ) ) )
        # data.extend(self._summaries)
        # return data 
        return self.pop()
    
    def reset(self):
        # self._transitions = {_var: 0 for _var in self._task_vars}
        # self._reset_data()
        return 

class MultiTaskAccumulator(StatAccumulator):

    def __init__(self, task_names,
                 eval_video_fps: int = 30, mean_only: bool = True,
                 train_prefix: str = 'train_',
                 eval_prefix: str = 'eval_'):
        self._train_accs = [_SimpleAccumulator(
            '%s%s/envs' % (train_prefix, name), eval_video_fps, mean_only=mean_only)
            for name in task_names ]
        self._eval_accs = [_SimpleAccumulator(
            '%s%s/envs' % (eval_prefix, name), eval_video_fps, mean_only=mean_only)
            for name in task_names]

        self._train_accs_mean = _SimpleAccumulator(
            '%ssummary/envs' % train_prefix, eval_video_fps,
            mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        replay_index = transition.info["task_id"]
        if eval:
            self._eval_accs[replay_index].step(transition, eval)
        else:
            self._train_accs[replay_index].step(transition, eval)
            self._train_accs_mean.step(transition, eval)

    def pop(self) -> List[Summary]:
        combined = self._train_accs_mean.pop()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.pop())
        return combined

    def peak(self) -> List[Summary]:
        combined = self._train_accs_mean.peak()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.peak())
        return combined

    def reset(self) -> None:
        self._train_accs_mean.reset()
        [acc.reset() for acc in self._train_accs + self._eval_accs]


class MultiTaskAccumulatorV2(StatAccumulator):
    """ Make sure each task's performance is pooled over all the variations 
        Stat not just different tasks but all their variations """
    def __init__(self, 
                 task_names: List[str],
                 tasks_vars: List[List[str]], # [ [task_A_0, task_A_1,...], [task_B_0, ...] ]
                 eval_video_fps: int = 30, 
                 mean_only: bool = True,
                 max_len: int = 5,
                 train_prefix: str = 'train',
                 eval_prefix: str = 'eval'
                 ):

        self._train_accs, self._eval_accs = [], [] 
        for i, (task_name, its_vars) in enumerate( zip(task_names, tasks_vars) ):
            self._train_accs.append(
                SimpleMultiVariationAccumulator(
                    train_prefix, task_name, i, its_vars, eval_video_fps, mean_only, max_len, TASK_ID, VAR_ID)
            )

            self._eval_accs.append(
                SimpleMultiVariationAccumulator(
                    eval_prefix, task_name, i, its_vars, eval_video_fps, mean_only, max_len, TASK_ID, VAR_ID)
            )
          
        self._train_accs_mean = _SimpleAccumulator(
            '%s_summary/all_tasks' % train_prefix, eval_video_fps,
            mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        replay_index = transition.info["task_id"]
        if eval:
            self._eval_accs[replay_index].step(transition, eval)
        else:
            self._train_accs[replay_index].step(transition, eval)
            self._train_accs_mean.step(transition, eval)

    def pop(self) -> List[Summary]:
        combined = self._train_accs_mean.pop()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.pop())
        return combined

    def peak(self) -> List[Summary]:
        combined = self._train_accs_mean.peak()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.peak())
        return combined

    def reset(self) -> None:
        self._train_accs_mean.reset()
        [acc.reset() for acc in self._train_accs + self._eval_accs]


    