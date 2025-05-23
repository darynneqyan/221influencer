"""
Main driver script for the Influencer MDP project.
Handles data loading, model training, and evaluation.
"""

import pandas as pd
from baselines import GreedyBaseline, RandomBaseline
from mdp import ValueIterationMDP
from utils import load_train_test_data

import json

def main():
    train_data, _ = load_train_test_data()
    train_data['engagement_rate'] = train_data['engagement'] / train_data['followers'].clip(lower=1)
    train_data = train_data.head(12)


    # Initialize and evaluate baselines
    greedy = GreedyBaseline(train_data)
    random = RandomBaseline(train_data)

     # Evaluate baselines
    greedy_result = greedy.evaluate()
    random_result = random.evaluate()

    print("\n--- Greedy Baseline ---")
    print(f"Total Engagement: {greedy_result['total_engagement']}")
    print("Influencers:", [step['username'] for step in greedy_result['selected_influencers']])

    print("\n--- Random Baseline ---")
    print(f"Total Engagement: {random_result['total_engagement']}")
    print("Influencers:", [step['username'] for step in random_result['selected_influencers']])


    # Initialize and evaluate MDP model
    mdp = ValueIterationMDP(train_data)
    mdp_result = mdp.evaluate()

    # Print MDP output
    print("\n--- MDP Step-by-Step Selection ---")
    for step in mdp_result['selected_influencers']:
        print(f"Step {step['step']}:")
        print(f"  Influencer: {step['username']}")
        print(f"  Cost: {step['cost']}")
        print(f"  Engagement: {step['engagement']}")
        print(f"  Remaining Budget Before: {step['remaining_budget']}\n")

    print(f"Total Engagement: {mdp_result['total_engagement']}")

if __name__ == "__main__":
    main()