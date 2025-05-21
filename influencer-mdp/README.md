# Influencer MDP

A Markov Decision Process (MDP) implementation for optimal influencer selection under budget constraints.

## Project Structure

```
influencer-mdp/
├── data/
│   └── influencers.csv           # preprocessed from Google Sheet
├── src/
│   ├── main.py                   # main driver script
│   ├── baselines.py              # greedy + random baselines
│   ├── mdp.py                    # value iteration model
│   ├── utils.py                  # preprocessing, reward calc
├── notebooks/
│   └── EDA.ipynb                 # initial data exploration
├── results/
│   └── baseline_eval.json
├── README.md
└── requirements.txt
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your influencer data in `data/influencers.csv` with the following columns:
   - followers: number of followers
   - engagements: number of engagements
   - cost: cost per post
   - (optional) engagement_rate: engagement rate

2. Run the main script:
```bash
python src/main.py
```

3. View results in `results/baseline_eval.json`

## Features

- Value Iteration MDP for optimal influencer selection
- Greedy and Random baselines for comparison
- Data preprocessing and reward calculation utilities
- Exploratory data analysis notebook

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- jupyter (for notebooks) 