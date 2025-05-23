"""
Utility functions for data preprocessing and reward calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def load_data(filepath: str = '../data/influencers.csv') -> pd.DataFrame:
    """
    Load and preprocess influencer data from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame with influencer data
    """
    df = pd.read_csv(filepath)
    
    # Basic preprocessing
    df = df.dropna()
    
    # Calculate engagement if not present
    if 'engagement' not in df.columns:
        df['engagement'] = df['likes'].fillna(0) + 2 * df['comments'].fillna(0) + 3 * df['saves'].fillna(0)
    # Calculate engagement rate if not present
    if 'engagement_rate' not in df.columns:
        df['engagement_rate'] = df['engagement'] / df['followers']
    
    # Normalize costs to reasonable range
    # TO DO: figure out a budget/cost structure that makes sense 
    if 'cost' in df.columns:
        df['cost'] = df['cost'].clip(lower=0)
    
    return df

# def calculate_rewards(influencers: List[Dict], 
#                      weights: Dict[str, float] = None) -> float:
#     """
#     Calculate total reward for a set of selected influencers.
    
#     Args:
#         influencers: List of selected influencer data
#         weights: Optional weights for different metrics
        
#     Returns:
#         Total reward score
#     """
#     if weights is None:
#         weights = {
#             'engagement_rate': 0.6,
#             'followers': 0.2,
#             'authenticity': 0.2
#         }
    
#     total_reward = 0
#     for influencer in influencers:
#         reward = 0
#         for metric, weight in weights.items():
#             if metric in influencer:
#                 reward += influencer[metric] * weight
#         total_reward += reward
        
#     return total_reward

# def preprocess_influencer_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Additional preprocessing steps for influencer data.
    
#     Args:
#         df: Raw influencer DataFrame
        
#     Returns:
#         Preprocessed DataFrame
#     """
#     # Remove outliers
#     for col in ['followers', 'engagements', 'cost']:
#         if col in df.columns:
#             q1 = df[col].quantile(0.25)
#             q3 = df[col].quantile(0.75)
#             iqr = q3 - q1
#             df = df[
#                 (df[col] >= q1 - 1.5 * iqr) & 
#                 (df[col] <= q3 + 1.5 * iqr)
#             ]
    
#     # Normalize numerical columns
#     for col in ['followers', 'engagements', 'cost']:
#         if col in df.columns:
#             df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
#     return df 

def load_train_test_data(train_path: str = '../data/train.csv', test_path: str = '../data/test.csv') -> tuple:
    # Load train + test
    train_df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='latin1')

    # Engagement = likes + 2*comments + 3*saves
    train_df['engagement'] = (
        train_df['likes'].fillna(0) +
        2 * train_df['comments'].fillna(0) +
        3 * train_df['saves'].fillna(0)
    )

    train_df['cost'] = train_df['engagement'] / 35
    train_df['cost'] = train_df['cost'].clip(lower=50, upper=350)

    train_df['engagement_rate'] = train_df['engagement'] / train_df['followers'].clip(lower=1)

    return train_df, test_df

