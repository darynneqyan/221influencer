# """
# Markov Decision Process implementation for influencer selection.
# Uses value iteration to find optimal policy.
# """
import numpy as np
from typing import List, Tuple, Dict
import math

class ValueIterationMDP:
    def __init__(self, data, gamma: float = 0.9, epsilon: float = 0.01, horizon: int = 3):
        self.data = data.to_dict('records')  # list of dicts with influencer info
        self.num_influencers = len(self.data)
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon

        self.states = self._initialize_states()
        self.policy = {}
        self.V = {}
        self.run_value_iteration()

    def _initialize_states(self) -> List[Tuple[int, Tuple[int], int]]:
        """
        Each state is a tuple of:
        - remaining budget (discretized)
        - a binary tuple of influencers already selected
        - the current step (0, 1, 2)
        """
        max_budget = 1000
        budget_levels = range(0, max_budget + 1, 100)
        step_range = range(self.horizon + 1)
        states = []

        for step in step_range:
            for budget in budget_levels:
                for used in range(2 ** self.num_influencers):
                    used_tuple = tuple((used >> i) & 1 for i in range(self.num_influencers))
                    states.append((budget, used_tuple, step))

        return states

    def _get_valid_actions(self, state) -> List[int]:
        """
        Return all influencers that:
        - haven't been used yet
        - cost <= remaining budget
        """
        budget, used_tuple, _ = state
        return [
            i for i, inf in enumerate(self.data)
            if used_tuple[i] == 0 and inf['cost'] <= budget
        ]



    def _get_reward(self, action: int) -> float:
        influencer = self.data[action]
        raw_score = influencer.get('likes', 0) + 2 * influencer.get('comments', 0) + 3 * influencer.get('saves', 0)
        return math.log(1 + raw_score)  # log-scaling to reduce extreme spikes



    def _get_transition(self, state: Tuple[int, Tuple[int], int], action: int) -> Tuple[int, Tuple[int], int]:
        """
        Apply action: subtract cost, mark influencer as used, move to next step
        """
        budget, used_tuple, step = state
        inf = self.data[action]
        new_budget = budget - inf['cost']
        new_used = list(used_tuple)
        new_used[action] = 1
        return (new_budget, tuple(new_used), step + 1)

    def run_value_iteration(self):
        """
        Classic value iteration over all states
        """
        for state in self.states:
            self.V[state] = 0.0

        while True:
            delta = 0
            new_V = self.V.copy()
            for state in self.states:
                budget, used_tuple, step = state
                if step == self.horizon or budget <= 0:
                    continue

                max_value = float('-inf')
                best_action = None
                for action in self._get_valid_actions(state):
                    reward = self._get_reward(action)
                    next_state = self._get_transition(state, action)
                    value = reward + self.gamma * self.V.get(next_state, 0)
                    if value > max_value:
                        max_value = value
                        best_action = action

                if best_action is not None:
                    new_V[state] = max_value
                    self.policy[state] = best_action
                    delta = max(delta, abs(self.V[state] - max_value))

            self.V = new_V
            if delta < self.epsilon:
                break

    def evaluate(self):
        """
        Start from initial state (full budget, nobody used),
        and return which influencers were selected + total engagement.
        """
        init_state = (1000, tuple([0] * self.num_influencers), 0)
        state = init_state
        total_reward = 0
        selected = []
        steps = []

        for _ in range(self.horizon):
            action = self.policy.get(state)
            if action is None:
                break
            inf = self.data[action]
            reward = self._get_reward(action)
            total_reward += reward
            steps.append({
                'step': len(selected),
                'username': inf.get('username'),
                'cost': inf.get('cost'),
                'engagement': reward,
                'remaining_budget': state[0],
                'niche': inf.get('niche'),
                'race': inf.get('race'),
                'gender': inf.get('gender')
            })
            selected.append(action)
            state = self._get_transition(state, action)

        return {
            'total_engagement': total_reward,
            'selected_influencers': steps
        }
