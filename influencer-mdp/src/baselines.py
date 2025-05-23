"""
Baseline implementations for the Influencer MDP project.
Selects a set of influencers in a single batch, given a fixed budget. Does not model sequential selection over multiple campaigns. 
Includes greedy and random selection strategies.
"""

import numpy as np
import random
import math

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
        """Select the single best influencer based on engagement rate per cost."""
        df = self.data.copy()
        df['engagement_rate_per_cost'] = df['engagement_rate'] / df['cost']
        sorted_influencers = df.sort_values('engagement_rate_per_cost', ascending=False)
        
        # Select only the best influencer within budget
        for _, influencer in sorted_influencers.iterrows():
            if influencer['cost'] <= self.budget:
                return [influencer]
        return []
    
    def evaluate(self):
        selected = self.select_influencers()
        return {
            'selected_influencers': selected,
            'total_engagement': sum(
                inf['likes'] + 2 * inf['comments'] + 3 * inf['saves']
                for inf in selected
            )
        }


class RandomBaseline:
    def __init__(self, data, budget=1000, engagement_cols=None):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        
    def select_influencers(self):
        """Select a single random influencer within budget."""
        df = self.data.copy()
        available = df[df['cost'] <= self.budget].copy()
        if len(available) == 0:
            return []
            
        # Shuffle the data to ensure true randomness
        available = available.sample(frac=1, random_state=None)  # No fixed seed
        
        # Take the first one that fits the budget
        for _, influencer in available.iterrows():
            if influencer['cost'] <= self.budget:
                return [influencer]
        return []
    
    def evaluate(self):
        selected = self.select_influencers()
        return {
            'selected_influencers': selected,
            'total_engagement': sum(
                inf['likes'] + 2 * inf['comments'] + 3 * inf['saves']
                for inf in selected
            )
        }