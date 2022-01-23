"""An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .uniform_replay_buffer import *
from .sum_tree import *
import numpy as np


PRIORITY = 'priority'


class PrioritizedReplayBuffer(UniformReplayBuffer):
    """An out-of-graph Replay Buffer for Prioritized Experience Replay.

    See uniform_replay_buffer.py for details.
    """

    def __init__(self, *args, **kwargs):
        """Initializes OutOfGraphPrioritizedReplayBuffer."""
        super(PrioritizedReplayBuffer, self).__init__(*args, **kwargs)
        self._sum_tree = SumTree(self._replay_capacity)
        self._reward_mean = 0.0 
        self._reward_std = 1.0 
        self._demo_cursor = 0 

    def get_storage_signature(self) -> Tuple[List[ReplayElement],
                                             List[ReplayElement]]:
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          dict of ReplayElements defining the type of the contents stored.
        """
        storage_elements, obs_elements = super(
            PrioritizedReplayBuffer, self).get_storage_signature()
        storage_elements.append(ReplayElement(PRIORITY, (), np.float32),)

        return storage_elements, obs_elements

    def add(self, action, reward, terminal, timeout, priority=None, **kwargs):
        kwargs['priority'] = priority
        super(PrioritizedReplayBuffer, self).add(
            action, reward, terminal, timeout, **kwargs)

    def _add(self, kwargs: dict):
        """Internal add method to add to the storage arrays.

        Args:
          kwargs: All the elements in a transition.
        """
        with self._lock:
            cursor = self.cursor()
            priority = kwargs[PRIORITY]
            if priority is None:
                priority = self._sum_tree.max_recorded_priority

            if self._disk_saving:
                self._store[TERMINAL][cursor] = kwargs[TERMINAL]
                with open(join(self._save_dir, '%d.replay' % cursor), 'wb') as f:
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                if self._add_count == 0:
                    self._add_initial_to_disk(kwargs)
            else:
                for name, data in kwargs.items():
                    self._store[name][cursor] = data

            self._sum_tree.set(self.cursor(), priority)
            self._add_count += 1
            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)
    
    def get_reward_stats(self):
        with self._lock:
          cursor = self.cursor()
          self._reward_mean = np.mean(self._store[REWARD][:cursor])
          if np.isinf(self._reward_mean) or np.isnan(self._reward_mean):
            self._reward_mean = 0
          self._reward_std  = np.std(self._store[REWARD][:cursor])
          if np.isinf(self._reward_std) or np.isnan(self._reward_std):
            self._reward_std = 0
        return self._reward_mean, self._reward_std 

    def add_final(self, **kwargs):
        """Adds a transition to the replay memory.
        Args:
          **kwargs: The remaining args
        """
        if self.is_empty() or self._store['terminal'][self.cursor() - 1] != 1:
            raise ValueError('The previous transition was not terminal.')
        self._check_add_types(kwargs, self._obs_signature)
        transition = self._final_transition(kwargs)
        for element_type in self._storage_signature:
            # 0 priority for final observation.
            if element_type.name == PRIORITY:
                transition[element_type.name] = 0.0
        self._add(transition)

    def update_demo_cursor(self):
      self._demo_cursor = self.cursor()

    def context_avaliable(self, batch_size, window_size):
      if self._add_count < window_size:
        return False
      if self.cursor() - self._demo_cursor < window_size:
        return False 
      return True 

    def sample_recent_batch(self, batch_size, window_size, pack_in_dict=True):
      """
      First sample indices from the most recent window of transitions, hence "loosely" on-policy data,
      then do the get the actual samples according to the indices. 
      Don't touch the demo transitions 
      """
      assert self._add_count >= window_size, 'Not enough data in replay buffer'
      assert self.cursor() - self._demo_cursor >= batch_size, 'Need agent-generated data for context sampling'
      
      low_bound = self.cursor() - window_size
      if self.cursor() - self._demo_cursor < window_size:
        low_bound = self.cursor() - batch_size
      recent_window = list(range(low_bound, self.cursor()))
      indices = np.random.choice(recent_window, batch_size)
      allowed_attempts = self._max_sample_attempts
      for i in range(len(indices)):
          if not self.is_valid_transition(indices[i]):
              if allowed_attempts == 0:
                  raise RuntimeError(
                      'Max sample attempts: Tried {} times but only sampled {}'
                      ' valid indices. Batch size is {}'.
                          format(self._max_sample_attempts, i, batch_size))
              index = indices[i]
              while not self.is_valid_transition(index) and allowed_attempts > 0:
                  # If index i is not valid keep sampling others. Note that this
                  # is not stratified.
                  index = np.random.choice(recent_window, 1)[0]
                  allowed_attempts -= 1
              indices[i] = index

      transition = super(PrioritizedReplayBuffer, self).sample_transition_batch(
        batch_size, indices, pack_in_dict=False)

      transition_elements = self.get_transition_elements(batch_size)
      transition_names = [e.name for e in transition_elements]
      probabilities_index = transition_names.index('sampling_probabilities')
      indices_index = transition_names.index('indices')
      indices = transition[indices_index]
      # The parent returned an empty array for the probabilities. Fill it with the
      # contents of the sum tree.
      transition[probabilities_index][:] = self.get_priority(indices)
      batch_arrays = transition
      if pack_in_dict:
          batch_arrays = self.unpack_transition(transition, transition_elements)
      return batch_arrays
 

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled as in Schaul et al. (2015).

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          Exception: If the batch was not constructed after maximum number of tries.
        """
        # Sample stratified indices. Some of them might be invalid.
        indices = self._sum_tree.stratified_sample(batch_size)
        # print('replay buffer sampling:', indices)
        allowed_attempts = self._max_sample_attempts
        for i in range(len(indices)):
            if not self.is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                            format(self._max_sample_attempts, i, batch_size))
                index = indices[i]
                while not self.is_valid_transition(
                        index) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self._sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index
        return indices

    def sample_transition_batch(self, batch_size=None, indices=None,
                                pack_in_dict=True):
        """Returns a batch of transitions with extra storage and the priorities.

        The extra storage are defined through the extra_storage_types constructor
        argument.

        When the transition is terminal next_state_batch has undefined contents.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        """
        transition = super(
            PrioritizedReplayBuffer, self).sample_transition_batch(
            batch_size, indices, pack_in_dict=False)

        transition_elements = self.get_transition_elements(batch_size)
        transition_names = [e.name for e in transition_elements]
        probabilities_index = transition_names.index('sampling_probabilities')
        indices_index = transition_names.index('indices')
        indices = transition[indices_index]
        # The parent returned an empty array for the probabilities. Fill it with the
        # contents of the sum tree.
        transition[probabilities_index][:] = self.get_priority(indices)
        batch_arrays = transition
        if pack_in_dict:
            batch_arrays = self.unpack_transition(transition,
                                                  transition_elements)
        return batch_arrays

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
          priorities: float, the corresponding priorities.
        """
        # print('Setting priorities:', indices, priorities)
        assert indices.dtype == np.int32, ('Indices must be integers, '
                                           'given: {}'.format(indices.dtype))
        assert not np.any(np.isnan(priorities)), 'Got dtype NaN in incomiong priority values'                   
        for index, priority in zip(indices, priorities):
            self._sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.

        For any memory location not yet used, the corresponding priority is 0.

        Args:
          indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).

        Returns:
          priorities: float, the corresponding priorities.
        """
        assert indices.shape, 'Indices must be an array.'
        assert indices.dtype == np.int32, ('Indices must be int32s, '
                                           'given: {}'.format(indices.dtype))
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self._sum_tree.get(memory_index)
        return priority_batch

    def get_average_priority(self):
        return self._sum_tree._total_priority() / max(self._add_count, 1) 

    def get_max_priority(self):
      return self._sum_tree.max_recorded_priority

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        parent_transition_type = (
            super(PrioritizedReplayBuffer,
                  self).get_transition_elements(batch_size))
        probablilities_type = [
            ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        return parent_transition_type + probablilities_type
