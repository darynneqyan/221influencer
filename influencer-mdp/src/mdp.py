"""
Markov Decision Process implementation for influencer selection.
Uses value iteration to find optimal policy.
"""

import numpy as np
from typing import List, Dict, Tuple

class ValueIterationMDP:
    def __init__(self, data, gamma: float = 0.9, epsilon: float = 0.01):
        self.data = data
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # convergence threshold
        self.states = self._initialize_states()
        self.actions = self._initialize_actions()
        self.transitions = self._initialize_transitions()
        self.rewards = self._initialize_rewards()
        
    def _initialize_states(self) -> List[Tuple]:
        """Initialize state space."""
        # States are (remaining_budget, selected_influencers)
        return [(budget, tuple()) for budget in range(0, 1001, 100)]
    
    def _initialize_actions(self) -> List[int]:
        """Initialize action space."""
        # Actions are indices of influencers to select
        return list(range(len(self.data)))
    
    def _initialize_transitions(self) -> Dict:
        """Initialize transition probabilities."""
        # For deterministic MDP, transitions are 1.0
        return {state: {action: 1.0 for action in self.actions} 
                for state in self.states}
    
    def _initialize_rewards(self) -> Dict:
        """Initialize reward function."""
        rewards = {}
        for state in self.states:
            budget, selected = state
            rewards[state] = {}
            for action in self.actions:
                influencer = self.data.iloc[action]
                if influencer['cost'] <= budget:
                    rewards[state][action] = influencer['engagement_rate']
                else:
                    rewards[state][action] = -float('inf')
        return rewards
    
    def value_iteration(self) -> Dict:
        """Perform value iteration to find optimal policy."""
        V = {state: 0 for state in self.states}
        policy = {state: None for state in self.states}
        
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                # Find best action
                best_value = float('-inf')
                best_action = None
                
                for action in self.actions:
                    if self.rewards[state][action] == float('-inf'):
                        continue
                        
                    next_state = self._get_next_state(state, action)
                    value = self.rewards[state][action] + self.gamma * V[next_state]
                    
                    if value > best_value:
                        best_value = value
                        best_action = action
                
                V[state] = best_value
                policy[state] = best_action
                delta = max(delta, abs(v - V[state]))
            
            if delta < self.epsilon:
                break
                
        return policy
    
    def _get_next_state(self, state: Tuple, action: int) -> Tuple:
        """Get next state given current state and action."""
        budget, selected = state
        influencer = self.data.iloc[action]
        new_budget = budget - influencer['cost']
        new_selected = selected + (action,)
        return (new_budget, new_selected)
    
    def evaluate(self) -> Dict:
        """Evaluate the MDP policy."""
        policy = self.value_iteration()
        initial_state = (1000, tuple())  # Start with full budget
        current_state = initial_state
        selected = []
        total_reward = 0
        
        while True:
            action = policy[current_state]
            if action is None:
                break
                
            selected.append(self.data.iloc[action])
            total_reward += self.rewards[current_state][action]
            current_state = self._get_next_state(current_state, action)
            
        return {
            'total_engagement': total_reward,
            'total_cost': sum(x['cost'] for x in selected),
            'num_selected': len(selected)
        } 