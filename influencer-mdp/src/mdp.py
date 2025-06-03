import numpy as np
from typing import List, Tuple, Dict
import pandas as pd

StateType = Tuple[int, int, int]

class ValueIterationMDP:
    def __init__(self, data, gamma: float = 0.9, epsilon: float = 0.01, horizon: int = 3,
                 engagement_weights: Dict[str, float] = None,
                 underrepresented_groups: List[str] = None,
                 diversity_first_selection_bonus: float = 1000.0,
                 debug_mode: bool = False):
        """
        ValueIterationMDP for influencer selection.
        State: (current_budget, current_step, has_selected_underrepresented)
        Engagement formula: (likes * w1) + (comments * w2) + (saves * w3)
        Weights are configurable via engagement_weights.
        Bonus applied for the first selection of an underrepresented influencer.
        """
        self.data = data.to_dict('records')
        self.num_influencers = len(self.data)
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon
        self.debug_mode = debug_mode
        self.engagement_weights = engagement_weights or {
            'likes': 1.0,
            'comments': 2.0,
            'saves': 3.0
        }
        if self.debug_mode: print(f"[MDP] Using engagement weights: {self.engagement_weights}")

        # Default underrepresented groups/diversity bonus
        self.underrepresented_groups = underrepresented_groups or ['Black', 'Latinx']
        self.diversity_first_selection_bonus = diversity_first_selection_bonus
        if self.debug_mode: print(f"[MDP] Using underrepresented groups: {self.underrepresented_groups} with first selection bonus {self.diversity_first_selection_bonus}")

        self.states = self._initialize_states()
        if self.debug_mode: print(f"[MDP] Initialized {len(self.states)} states.")
        self.V = {}
        self.policy = {}

        self.run_value_iteration()

    def _initialize_states(self) -> List[StateType]:
        max_budget = 1000
        budget_levels = range(0, max_budget + 1, 10)
        states: List[StateType] = []

        for step in range(self.horizon + 1):
            for budget in budget_levels:
                states.append((budget, step, 0))      # has_selected_underrepresented = 0
                if step > 0:
                     states.append((budget, step, 1)) # has_selected_underrepresented = 1

        return states

    def _get_valid_actions(self, state: StateType) -> List[int]:
        budget, step, has_selected_underrepresented = state
        valid_actions = []
        
        # Get all valid actions based on budget and not previously selected
        for i, inf in enumerate(self.data):
            # Check if we can afford this influencer
            if inf['cost'] <= budget:
                # Check if this influencer has already been selected
                if step == 0 or i not in self.policy.values():
                    valid_actions.append(i)
        
        # If we have valid actions + remaining budget is significant
        if valid_actions and budget > 100:
            # Sort actions by cost to prioritize using more budget
            valid_actions.sort(key=lambda i: self.data[i]['cost'], reverse=True)
            
        return valid_actions

    def _calculate_engagement(self, influencer: Dict) -> float:
        """
        Calculate engagement score based on configured weights.
        Assumes raw engagement has already been scaled.
        """
        base_score = sum(
            influencer.get(metric, 0) * weight 
            for metric, weight in self.engagement_weights.items()
        )

        engagement_rate = influencer.get('engagement_rate')
        if engagement_rate is not None and pd.notna(engagement_rate):
             base_score *= engagement_rate

        return base_score

    def _get_reward(self, state: StateType, action: int) -> float:
        """
        Reward function that considers engagement, budget usage, and diversity coverage.
        State: (current_budget, current_step, has_selected_underrepresented)
        """
        budget, step, has_selected_underrepresented = state
        inf = self.data[action]
        base_score = self._calculate_engagement(inf)
        
        # Add bonus for first selection of an underrepresented influencer
        is_underrepresented = inf.get('race') in self.underrepresented_groups
        if has_selected_underrepresented == 0 and is_underrepresented:
            base_score *= 3.0  # Triple the base score for first diverse selection
            base_score += self.diversity_first_selection_bonus
            if self.debug_mode: print(f"[MDP Reward] Added diversity first selection bonus {self.diversity_first_selection_bonus} and tripled base score for action {action} (race: {inf.get('race')}) at state {state}.")

        remaining_budget = budget - inf.get('cost', 0)
        
        # Penalize if leaves too much budget
        steps_remaining = self.horizon - step - 1
        if steps_remaining > 0 and remaining_budget > (steps_remaining * 200):
            final_reward = base_score * 0.5
            if self.debug_mode: print(f"[MDP Reward] Applied high remaining budget penalty (x0.5) for action {action} at state {state}. Remaining budget: {remaining_budget:.2f}, Steps remaining: {steps_remaining}")
            return final_reward
            
        # If leaves a good amount of budget, give small bonus
        if remaining_budget < 100 and remaining_budget >= 0:    # Ensures non-negative remaining
            final_reward = base_score * 1.2
            if self.debug_mode: print(f"[MDP Reward] Applied <100 budget bonus (x1.2) for action {action} at state {state}. Remaining budget: {remaining_budget:.2f}")
            return final_reward

        # If remaining budget is 0, give a bonus for perfect budget utilization
        if remaining_budget == 0:
             final_reward = base_score * 1.5                    # Bonus for perfect utilization
             if self.debug_mode: print(f"[MDP Reward] Applied ==0 budget bonus (x1.5) for action {action} at state {state}. Remaining budget: {remaining_budget:.2f}")
             return final_reward

        final_reward = base_score
        if self.debug_mode: print(f"[MDP Reward] Default reward for action {action} at state {state}. Remaining budget: {remaining_budget:.2f}")
        return final_reward

    def _get_transition(self, state: StateType, action: int) -> StateType:
        budget, step, has_selected_underrepresented = state
        inf = self.data[action]
        new_budget = budget - inf['cost']
        new_step = step + 1
        
        # Update has_selected_underrepresented flag
        is_underrepresented = inf.get('race') in self.underrepresented_groups
        new_has_selected_underrepresented = 1 if is_underrepresented else has_selected_underrepresented

        return (new_budget, new_step, new_has_selected_underrepresented)

    def run_value_iteration(self):
        for state in self.states:
            self.V[state] = 0.0

        iteration = 0
        while True:
            delta = 0
            new_V = self.V.copy()

            if self.debug_mode: print(f"\n--- Value Iteration Iteration {iteration} ---\n")

            # Ensure budget levels list is accessible
            budget_levels = sorted(list(set(state[0] for state in self.states)))

            for state in self.states:
                budget, step, has_selected_underrepresented = state

                # Terminal states
                if step >= self.horizon or budget <= 0:
                    # Consistent terminal reward calculation
                    terminal_reward = 0
                    
                    # Severe penalty for not selecting enough influencers
                    if step < self.horizon:
                        terminal_reward -= 10000000.0  # Much larger penalty for not selecting enough influencers
                        if self.debug_mode: print(f"  State {state}: Terminal state with insufficient selections, Value: {terminal_reward:.2f}")
                    
                    # Larger bonus if diverse creator was selected
                    if has_selected_underrepresented == 1:
                        terminal_reward += 5000.0
                    
                    # Penalty for leaving too much budget at the end of horizon
                    if step >= self.horizon and budget > 100:
                        terminal_reward -= budget * 0.1
                    
                    new_V[state] = terminal_reward
                    if self.debug_mode: print(f"  State {state}: Terminal state, Value: {new_V[state]:.2f}")
                    continue

                max_value = float('-inf')
                best_action = None

                if self.debug_mode: print(f"  Processing state {state}:")

                for action in self._get_valid_actions(state):
                    reward = self._get_reward(state, action)
                    next_state_raw = self._get_transition(state, action)
                    
                    next_budget_raw = next_state_raw[0]
                    closest_next_budget = min(budget_levels, key=lambda x: abs(x - next_budget_raw))
                    next_state_lookup: StateType = (closest_next_budget, next_state_raw[1], next_state_raw[2])

                    value = reward + self.gamma * self.V.get(next_state_lookup, 0)

                    if self.debug_mode: print(f"    Action {action}: Reward {reward:.2f}, Next State (raw) {next_state_raw}, Next State (lookup) {next_state_lookup}, Next State Value {self.V.get(next_state_lookup, 0):.2f}, Total Value {value:.2f}")

                    if value > max_value:
                        max_value = value
                        best_action = action

                if best_action is not None: 
                    if self.debug_mode: print(f"  Best Action for {state}: {best_action}, Max Value: {max_value:.2f}, Previous Value: {self.V[state]:.2f}")
                    new_V[state] = max_value
                    self.policy[state] = best_action
                    delta = max(delta, abs(self.V[state] - max_value))
                elif self.debug_mode: 
                    print(f"  No valid action for state {state}. Value remains {self.V[state]:.2f}")

            self.V = new_V
            iteration += 1
            if delta < self.epsilon:
                if self.debug_mode: print(f"Value Iteration converged after {iteration} iterations with delta {delta:.4f}.")
                break
            if iteration > 500:
                if self.debug_mode: print(f"Value Iteration did not converge after 500 iterations. Delta: {delta:.4f}")
                break

    def evaluate(self):
        # Initial state includes budget, step (0), and has_selected_underrepresented (0)
        init_state: StateType = (1000, 0, 0)
        state: StateType = init_state
        total_evaluated_reward = 0
        selected = []
        steps = []
        total_cost = 0

        print(f"[MDP Evaluate] Starting evaluation from state: {init_state}")
        budget_levels = sorted(list(set(state[0] for state in self.states)))

        while True:
            print(f"[MDP Evaluate] Current state: {state}")

            # Find the closest available budget level in the state space for policy lookup
            actual_budget = state[0]
            closest_budget = min(budget_levels, key=lambda x: abs(x - actual_budget))

            # Construct the state for policy lookup using the closest budget level + current diversity flag
            policy_lookup_state: StateType = (closest_budget, state[1], state[2])

            print(f"[MDP Evaluate] Looking up policy for state: {policy_lookup_state} (Actual budget: {actual_budget:.2f})")
            action = self.policy.get(policy_lookup_state)
            print(f"[MDP Evaluate] Policy lookup result (action): {action}")

            # Check terminal conditions
            if action is None or state[1] >= self.horizon or state[0] <= 0:
                print("[MDP Evaluate] Stopping evaluation: Action is None, Horizon reached, or Budget depleted")
                break

            inf = self.data[action]
            
            current_step_reward = self._get_reward(state, action)

            total_evaluated_reward += current_step_reward
            total_cost += inf.get('cost', 0)

            raw_engagement = self._calculate_engagement(inf)

            steps.append({
                'step': len(selected),
                'username': inf.get('username'),
                'cost': inf.get('cost'),
                'raw_engagement': raw_engagement,
                'reward_applied': current_step_reward, 
                'remaining_budget': state[0],
                'niche': inf.get('niche'),
                'race': inf.get('race'),
                'gender': inf.get('gender'),
                'total_cost': total_cost
            })

            selected.append(action)
            state = self._get_transition(state, action)

        return {
            'total_engagement': total_evaluated_reward, 
            'total_cost': total_cost,
            'selected_influencers': steps
        }

    def get_baseline_comparison(self):
        """
        Compare MDP results with a simple baseline that selects influencers
        based on engagement/cost ratio
        """
        # Calculate engagement/cost ratio for each influencer
        influencer_ratios = []
        for i, inf in enumerate(self.data):
            engagement = self._calculate_engagement(inf)
            cost = inf.get('cost', 0)
            if cost > 0:
                ratio = engagement / cost
                influencer_ratios.append((i, ratio, engagement, cost))
    
        influencer_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Select influencers until budget exhausted
        budget = 1000
        baseline_selection = []
        total_cost = 0
        total_engagement = 0
        
        for i, ratio, engagement, cost in influencer_ratios:
            if total_cost + cost <= budget:
                baseline_selection.append({
                    'username': self.data[i].get('username'),
                    'cost': cost,
                    'engagement': engagement,
                    'ratio': ratio
                })
                total_cost += cost
                total_engagement += engagement
        
        return {
            'baseline_selection': baseline_selection,
            'total_cost': total_cost,
            'total_engagement': total_engagement
        }
