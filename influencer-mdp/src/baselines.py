"""
Baseline implementations for the Influencer MDP project.
Includes greedy and random selection strategies.
"""

import numpy as np
import random

class GreedyBaseline:
    def __init__(self, data, budget=1000, engagement_cols=None):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        
    def select_influencers(self):
        """Select influencers greedily based on engagement per cost.
        engagement = likes + 2 * comments + 3 * saves """
        df = self.data.copy()
        df['engagement'] = df['likes'].fillna(0) + 2 * df['comments'].fillna(0) + 3 * df['saves'].fillna(0)
        df['engagement_per_cost'] = df['engagement'] / df['cost']
        sorted_influencers = df.sort_values('engagement_per_cost', ascending=False)
        selected = []
        remaining_budget = self.budget
        
        for _, influencer in sorted_influencers.iterrows():
            if influencer['cost'] <= remaining_budget:
                selected.append(influencer)
                remaining_budget -= influencer['cost']
        return selected
    
    def evaluate(self):
        selected = self.select_influencers()
        total_engagement = sum(x['engagement'] for x in selected)
        total_cost = sum(x['cost'] for x in selected)
        diversity = len(set(x['username'] for x in selected)) if len(selected) > 0 and 'username' in selected[0] else len(selected)
        budget_utilization = total_cost / self.budget if self.budget else 0
        return {
            'total_engagement': total_engagement,
            'total_cost': total_cost,
            'num_selected': len(selected),
            'diversity': diversity,
            'budget_utilization': budget_utilization
        }

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
        total_engagement = sum(
            x['likes'] + 2 * x['comments'] + 3 * x['saves'] for _, x in selected.iterrows()
        )
        total_cost = sum(x['cost'] for _, x in selected.iterrows())
        diversity = len(set(x['username'] for _, x in selected.iterrows())) if not selected.empty and 'username' in selected.columns else len(selected)
        budget_utilization = total_cost / self.budget if self.budget else 0
        return {
            'total_engagement': total_engagement,
            'total_cost': total_cost,
            'num_selected': len(selected),
            'diversity': diversity,
            'budget_utilization': budget_utilization
        } 