"""
Baseline implementations for the Influencer MDP project.
Selects a set of influencers in a single batch, given a fixed budget. Does not model sequential selection over multiple campaigns. 
Includes greedy and random selection strategies.
"""

import numpy as np
import pandas as pd

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
    def __init__(self, data, budget=1000, engagement_cols=None, horizon=3):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        self.horizon = horizon
        
    def select_influencers(self):
        """Select up to `horizon` best influencers based on engagement rate per cost, within budget."""
        df = self.data.copy()
        # Ensure engagement rate per cost is calculated + handle division by zero/NaN costs
        df['engagement_rate_per_cost'] = df.apply(lambda row: row['engagement_rate'] / row['cost'] if row['cost'] > 0 and not pd.isna(row['cost']) else 0, axis=1)
        sorted_influencers = df.sort_values('engagement_rate_per_cost', ascending=False)
        selected = []
        total_cost = 0
        for _, influencer in sorted_influencers.iterrows():
            cost = influencer['cost'] if not pd.isna(influencer['cost']) else 0
            if len(selected) >= self.horizon:
                break
            if total_cost + cost <= self.budget:
                selected.append(influencer)
                total_cost += cost
        return selected
    
    def evaluate(self):
        selected = self.select_influencers()
        
        # Calculate total engagement 
        total_engagement = sum(
            (inf.get('likes', 0) if pd.notna(inf.get('likes', 0)) else 0) + 
            2 * (inf.get('comments', 0) if pd.notna(inf.get('comments', 0)) else 0) + 
            3 * (inf.get('saves', 0) if pd.notna(inf.get('saves', 0)) else 0)
            for inf in selected
        )

        return {
            'selected_influencers': selected,
            'total_engagement': total_engagement
        }
        


class RandomBaseline:
    def __init__(self, data, budget=1000, engagement_cols=None, horizon=3):
        self.data = data
        self.budget = budget
        self.engagement_cols = engagement_cols or ['likes', 'comments', 'saves']
        self.horizon = horizon
        
    def select_influencers(self):
        """Select up to `horizon` random influencers within budget."""
        df = self.data.copy()
        # Ensure cost is a number before filtering
        df['cost'] = df['cost'].apply(lambda x: x if not pd.isna(x) else 0)
        available = df[df['cost'] <= self.budget].copy()
        if len(available) == 0:
            return []
        available = available.sample(frac=1, random_state=None)  # Shuffle
        selected = []
        total_cost = 0
        for _, influencer in available.iterrows():
            # Ensure cost is a number before adding
            cost = influencer['cost'] if not pd.isna(influencer['cost']) else 0
            if len(selected) >= self.horizon:
                break
            if total_cost + cost <= self.budget:
                selected.append(influencer)
                total_cost += cost
        return selected
    
    def evaluate(self):
        selected = self.select_influencers()
        
        # Calculate total engagement using the same logic as MDP's 
        total_engagement = sum(
            (inf.get('likes', 0) if pd.notna(inf.get('likes', 0)) else 0) + 
            2 * (inf.get('comments', 0) if pd.notna(inf.get('comments', 0)) else 0) + 
            3 * (inf.get('saves', 0) if pd.notna(inf.get('saves', 0)) else 0)
            for inf in selected
        )

        return {
            'selected_influencers': selected,
            'total_engagement': total_engagement
        }
