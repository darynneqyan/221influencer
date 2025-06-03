"""
Main driver script for the Influencer MDP project.
Handles data loading, model training, and evaluation.
"""

import pandas as pd
from baselines import GreedyBaseline, RandomBaseline
from mdp import ValueIterationMDP
from utils import load_train_test_data
import math
import numpy as np

# ---- CONFIGURABLE ENGAGEMENT WEIGHTS ----
ENGAGEMENT_WEIGHTS = {
    'likes': 1.0,
    'comments': 2.0,
    'saves': 3.0
}

# ---- CONFIGURABLE HORIZON AND BUDGET ----
HORIZON = 3
BUDGET = 1000

# ---- CONFIGURABLE DIVERSITY SETTINGS ----
UNDERREPRESENTED_GROUPS = ['Black', 'Latinx']
DIVERSITY_FIRST_SELECTION_BONUS = 500000.0 # Increased for impact

# Print the engagement formula and weights
print("\n[INFO] Engagement formula: (likes * {likes}) + (comments * {comments}) + (saves * {saves})".format(**ENGAGEMENT_WEIGHTS))
print(f"[INFO] Horizon: {HORIZON}, Budget: {BUDGET}")
print(f"[INFO] Underrepresented Groups for Diversity Bonus: {UNDERREPRESENTED_GROUPS} with bonus {DIVERSITY_FIRST_SELECTION_BONUS}")

def calculate_engagement(influencer):
    """Consistent engagement calculation across all methods"""
    return (
        influencer['likes'] + 
        2 * influencer['comments'] + 
        3 * influencer['saves']
    ) * influencer.get('engagement_rate', 1.0)

def main():
    train_data, _ = load_train_test_data()
    
    # Calculate engagement rate
    train_data['engagement_rate'] = train_data['engagement'] / train_data['followers'].clip(lower=1)
    
    # Limit to first 12 influencers for testing
    train_data = train_data.head(12)

    # Calculate base cost using a combination of engagement and followers
    # This creates more variation in costs
    train_data['base_cost'] = (
        (train_data['engagement'] / 1000) +  # Engagement component
        (train_data['followers'] / 10000)    # Follower component
    )

    # Scale the base cost to get a reasonable range
    train_data['cost'] = train_data['base_cost'] * 50

    # Ensure costs are within reasonable bounds and have more granularity
    train_data['cost'] = train_data['cost'].clip(lower=100, upper=400)
    train_data['cost'] = train_data['cost'].round(2)  # Round to 2 decimal places

    # Apply affirmative action: BOOST costs for underrepresented groups
    # This makes them more competitive in the selection process
    underrepresented_groups = ['Black', 'Latinx']
    train_data.loc[train_data['race'].isin(underrepresented_groups), 'cost'] *= 1.5  # 50% boost

    print("\n[INFO] Influencer costs after dynamic calculation and affirmative action:")
    print(train_data[['username', 'race', 'cost', 'engagement', 'followers']].sort_values('cost', ascending=False))

    # Initialize and evaluate baselines
    greedy = GreedyBaseline(train_data, budget=BUDGET, horizon=HORIZON)
    random = RandomBaseline(train_data, budget=BUDGET, horizon=HORIZON)

    # Evaluate baselines
    greedy_result = greedy.evaluate()
    random_result = random.evaluate()

    print("\n--- Greedy Baseline ---")
    print(f"Total Engagement: {greedy_result['total_engagement']}")
    print("Influencers:", [step['username'] for step in greedy_result['selected_influencers']])
    print(f"Total Cost: {sum(step['cost'] for step in greedy_result['selected_influencers'])}")
    print(f"Number of Influencers Selected: {len(greedy_result['selected_influencers'])}")

    print("\n--- Random Baseline ---")
    print(f"Total Engagement: {random_result['total_engagement']}")
    print("Influencers:", [step['username'] for step in random_result['selected_influencers']])
    print(f"Total Cost: {sum(step['cost'] for step in random_result['selected_influencers'])}")
    print(f"Number of Influencers Selected: {len(random_result['selected_influencers'])}")

    # Initialize and evaluate MDP model
    mdp = ValueIterationMDP(train_data, gamma=0.9, epsilon=0.01, horizon=HORIZON,
                            engagement_weights=ENGAGEMENT_WEIGHTS,
                            underrepresented_groups=UNDERREPRESENTED_GROUPS,
                            diversity_first_selection_bonus=DIVERSITY_FIRST_SELECTION_BONUS,
                            debug_mode=False)  # Disable debug mode
    mdp_result = mdp.evaluate()

    # Print MDP output
    print("\n--- MDP Step-by-Step Selection ---")
    for step in mdp_result['selected_influencers']:
        print(f"Step {step['step']}:")
        print(f"  Influencer: {step['username']}")
        print(f"  Cost: {step['cost']:.2f}")
        print(f"  Raw Engagement: {step['raw_engagement']:.2f}")
        print(f"  Reward (Applied in MDP): {step['reward_applied']:.2f}")
        print(f"  Engagement Rate: {train_data[train_data['username'] == step['username']]['engagement_rate'].iloc[0]:.4f}")
        print(f"  Remaining Budget Before: {step['remaining_budget']:.2f}\n")

    print(f"Total Engagement (sum of applied rewards): {mdp_result['total_engagement']:.2f}")
    print(f"Total Cost: {mdp_result['total_cost']:.2f}")
    print(f"Number of Influencers Selected: {len(mdp_result['selected_influencers'])}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    def fmt(val):
        if isinstance(val, float):
            if math.isnan(val):
                return 'NaN'
            return f"{val:.2f}"
        return str(val)
    print(f"{'Method':<20}{'Engagement':<15}{'Cost':<10}{'# Selected':<12}")
    print(f"{'Greedy':<20}{fmt(greedy_result['total_engagement']):<15}{fmt(sum(step['cost'] for step in greedy_result['selected_influencers'])):<10}{len(greedy_result['selected_influencers']):<12}")
    print(f"{'Random':<20}{fmt(random_result['total_engagement']):<15}{fmt(sum(step['cost'] for step in random_result['selected_influencers'])):<10}{len(random_result['selected_influencers']):<12}")
    print(f"{'MDP':<20}{fmt(mdp_result['total_engagement']):<15}{fmt(mdp_result['total_cost']):<10}{len(mdp_result['selected_influencers']):<12}")

    # Warn if NaN detected
    if any(math.isnan(x) for x in [greedy_result['total_engagement'], random_result['total_engagement'], mdp_result['total_engagement']]):
        print("[WARNING] NaN detected in engagement results! Check your data for missing values.")
    if any(math.isnan(x) for x in [sum(step['cost'] for step in greedy_result['selected_influencers']), sum(step['cost'] for step in random_result['selected_influencers']), mdp_result['total_cost']]):
        print("[WARNING] NaN detected in cost results! Check your data for missing values.")

    print(f"\n[INFO] Actual steps taken by MDP: {len(mdp_result['selected_influencers'])}")
    print(f"[INFO] Actual budget used by MDP: {fmt(mdp_result['total_cost'])}")

if __name__ == "__main__":
    main()