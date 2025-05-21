"""
Main driver script for the Influencer MDP project.
Handles data loading, model training, and evaluation.
"""

import pandas as pd
from baselines import GreedyBaseline, RandomBaseline
from mdp import ValueIterationMDP
from utils import load_data, calculate_rewards
import json

def main():
    # Load and preprocess data
    data = load_data()
    
    # Initialize and run baselines
    greedy_baseline = GreedyBaseline(data)
    random_baseline = RandomBaseline(data)
    
    # Initialize and run MDP
    mdp = ValueIterationMDP(data)
    
    # Evaluate and compare results
    results = {
        'greedy': greedy_baseline.evaluate(),
        'random': random_baseline.evaluate(),
        'mdp': mdp.evaluate()
    }
    
    # Save results
    with open('../results/baseline_eval.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 