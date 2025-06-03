# CS 221 Extra Credit: Influencer Selection MDP 🌟🌟

An intelligent influencer selection system using Markov Decision Processes (MDP) to optimize campaign outcomes while promoting diversity and fair representation.

## Team Members
- Coco Hernandez
- Tarini Mutreja
- Nicole Esibov
- Darynne Lee

## Overview

This project implements a Value Iteration MDP to select influencers for marketing campaigns. The system considers multiple factors including:
- Engagement metrics (likes, comments, saves)
- Budget constraints
- Diversity and representation
- Cost efficiency
- Campaign reach

## Key Features

- **Smart Selection**: Uses MDP to learn optimal selection policies
- **Underrepresented Promotion**: Implements affirmative action principles
- **Budget Optimization**: Efficient allocation of campaign budget
- **Fair Pricing**: Dynamic cost calculation with equity adjustments
- **Multiple Baselines**: Comparison with greedy and random selection strategies


### State Space
The MDP uses a three-dimensional state space:
- Current budget
- Current step
- Diversity coverage flag

### Reward Structure
The reward function considers:
- Engagement metrics (weighted combination of likes, comments, saves)
- Budget utilization
- Diversity bonuses
- Selection constraints

### Cost Calculation
Costs are dynamically calculated based on:
- Engagement metrics
- Follower count
- Affirmative action adjustments for underrepresented groups

## Configuration

Key parameters can be adjusted in `src/main.py`:

```python
# Budget and Horizon
BUDGET = 1000
HORIZON = 3

# Engagement Weights
ENGAGEMENT_WEIGHTS = {
    'likes': 1.0,
    'comments': 2.0,
    'saves': 3.0
}

# Diversity Settings
UNDERREPRESENTED_GROUPS = ['Black', 'Latinx']
DIVERSITY_FIRST_SELECTION_BONUS = 500000.0
```

## Output

The system provides:
- Step-by-step selection process
- Cost breakdown
- Engagement metrics
- Diversity coverage
- Comparison with baseline methods

## Future Improvements

- Add more sophisticated engagement metrics
- Implement machine learning for cost prediction
- Add support for different campaign types
- Enhance diversity metrics
- Add visualization tools
