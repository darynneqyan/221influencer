import numpy as np
from typing import List, Tuple, Dict
import math

class ValueIterationMDP:
    def __init__(self, data, gamma: float = 0.9, epsilon: float = 0.01, horizon: int = 3):
        self.data = data.to_dict('records')
        self.num_influencers = len(self.data)
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon

        self.states = self._initialize_states()
        self.V = {}
        self.policy = {}

        self.run_value_iteration()

    def _initialize_states(self) -> List[Tuple[int, Tuple[int], int]]:
        max_budget = 1000
        # Use very fine granularity for budget
        budget_levels = range(0, max_budget + 1, 10)
        states = []

        for step in range(self.horizon + 1):
            for budget in budget_levels:
                for used in range(2 ** self.num_influencers):
                    used_tuple = tuple((used >> i) & 1 for i in range(self.num_influencers))
                    states.append((budget, used_tuple, step))

        return states

    def _get_valid_actions(self, state) -> List[int]:
        budget, used_tuple, _ = state
        valid_actions = []
        
        # Get all valid actions
        for i, inf in enumerate(self.data):
            if used_tuple[i] == 0 and inf['cost'] <= budget:
                valid_actions.append(i)
        
        # If we have valid actions and remaining budget is significant
        if valid_actions and budget > 100:
            # Sort actions by cost to prioritize using more budget
            valid_actions.sort(key=lambda i: self.data[i]['cost'], reverse=True)
            
        return valid_actions

    def _get_reward(self, action: int, state: Tuple[int, Tuple[int], int]) -> float:
        """
        Reward function that considers both engagement and budget usage
        """
        budget, used_tuple, step = state
        inf = self.data[action]
        
        # Base engagement score
        raw_score = inf.get('likes', 0) + 2 * inf.get('comments', 0) + 3 * inf.get('saves', 0)
        
        # Calculate remaining budget after this action
        remaining_budget = budget - inf.get('cost', 0)
        
        # If this would leave too much budget, penalize but don't make impossible
        if remaining_budget > 200:
            return raw_score * 0.5
            
        # If this would leave a good amount of budget, give bonus
        if remaining_budget < 100:
            return raw_score * 2.0
            
        return raw_score

    def _get_transition(self, state: Tuple[int, Tuple[int], int], action: int) -> Tuple[int, Tuple[int], int]:
        budget, used_tuple, step = state
        inf = self.data[action]
        new_budget = budget - inf['cost']
        new_used = list(used_tuple)
        new_used[action] = 1
        return (new_budget, tuple(new_used), step + 1)

    def run_value_iteration(self):
        for state in self.states:
            self.V[state] = 0.0

        while True:
            delta = 0
            new_V = self.V.copy()

            for state in self.states:
                budget, used_tuple, step = state

                # Terminal states
                if step == self.horizon or budget <= 0:
                    # If we have significant budget left, penalize but don't make impossible
                    if budget > 100:
                        new_V[state] = -budget * 0.1
                    else:
                        new_V[state] = 0
                    continue

                max_value = float('-inf')
                best_action = None

                for action in self._get_valid_actions(state):
                    reward = self._get_reward(action, state)
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
        init_state = (1000, tuple([0] * self.num_influencers), 0)
        state = init_state
        total_reward = 0
        selected = []
        steps = []
        total_cost = 0

        while True:
            action = self.policy.get(state)
            if action is None or total_cost >= 1000:  # Budget limit
                break

            inf = self.data[action]
            reward = self._get_reward(action, state)
            total_reward += reward
            total_cost += inf.get('cost', 0)

            # Calculate raw engagement for display
            raw_engagement = inf.get('likes', 0) + 2 * inf.get('comments', 0) + 3 * inf.get('saves', 0)

            steps.append({
                'step': len(selected),
                'username': inf.get('username'),
                'cost': inf.get('cost'),
                'engagement': raw_engagement,
                'remaining_budget': state[0],
                'niche': inf.get('niche'),
                'race': inf.get('race'),
                'gender': inf.get('gender'),
                'total_cost': total_cost
            })

            selected.append(action)
            state = self._get_transition(state, action)

        return {
            'total_engagement': sum(step['engagement'] for step in steps),
            'total_cost': total_cost,
            'selected_influencers': steps
        }
