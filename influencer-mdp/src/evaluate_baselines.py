import json
from utils import load_train_test_data
from baselines import GreedyBaseline, RandomBaseline

# Load train and test data
_, test_df = load_train_test_data()

# Set budget (can be adjusted as needed)
budget = 1000

# Evaluate Greedy Baseline
greedy = GreedyBaseline(test_df, budget=budget)
greedy_result = greedy.evaluate()

# Evaluate Random Baseline
random = RandomBaseline(test_df, budget=budget)
random_result = random.evaluate()

# Save results
results = {
    'greedy': greedy_result,
    'random': random_result
}

with open('../results/baseline_eval.json', 'w') as f:
    json.dump(results, f, indent=4)

print('Baseline evaluation complete. Results saved to results/baseline_eval.json') 