"""
Baseline implementations for the Influencer MDP project.
Selects a set of influencers in a single batch, given a fixed budget. Does not model sequential selection over multiple campaigns. 
Includes greedy and random selection strategies.
"""

import numpy as np
import random

def evaluate_selection(selected, budget):
    total_engagement_rate = sum(x['engagement_rate'] for x in selected)
    total_cost = sum(x['cost'] for x in selected)
    diversity = len(set(x['username'] for x in selected)) if len(selected) > 0 and 'username' in selected[0] else len(selected)
    budget_utilization = total_cost / budget if budget else 0
    total_engagement_rate_per_cost = total_engagement_rate / total_cost if total_cost > 0 else 0
    return {
        'total_engagement_rate': total_engagement_rate,
        'total_cost': total_cost,
        'num_selected': len(selected),
        'diversity': diversity,
        'budget_utilization': budget_utilization,
        'total_engagement_rate_per_cost': total_engagement_rate_per_cost
    }

class GreedyBaseline:
    def __init__(self, data, budget=1000, engagement_cols=None):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        
    def select_influencers(self):
        """Select influencers greedily based on engagement rate per cost."""
        df = self.data.copy()
        df['engagement_rate_per_cost'] = df['engagement_rate'] / df['cost']
        sorted_influencers = df.sort_values('engagement_rate_per_cost', ascending=False)
        selected = []
        remaining_budget = self.budget
        for _, influencer in sorted_influencers.iterrows():
            if influencer['cost'] <= remaining_budget:
                selected.append(influencer)
                remaining_budget -= influencer['cost']
        return selected
    
    def evaluate(self):
        selected = self.select_influencers()
        return evaluate_selection(selected, self.budget)

class RandomBaseline:
    def __init__(self, data, budget=1000, engagement_cols=None, seed=42):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        self.seed = seed
        
    def select_influencers(self):
        df = self.data.copy()
        available = df[df['cost'] <= self.budget].copy()
        selected = []
        remaining_budget = self.budget
        rng = np.random.default_rng(self.seed)
        indices = list(available.index)
        rng.shuffle(indices)
        for idx in indices:
            influencer = available.loc[idx]
            if influencer['cost'] <= remaining_budget:
                selected.append(influencer)
                remaining_budget -= influencer['cost']
        return selected
    
    def evaluate(self):
        selected = self.select_influencers()
        return evaluate_selection(selected, self.budget) 