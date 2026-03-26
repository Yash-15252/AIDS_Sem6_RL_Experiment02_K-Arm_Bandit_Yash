# AIDS_Sem6_RL_Experiment02_KArmedBandit

## Aim
To implement and compare different Multi-Armed Bandit algorithms (Random Selection, ε-greedy, UCB) for ad click optimization problem.

## Problem Statement
An advertising company runs 10 different ads on a webpage. Given click data (1=clicked, 0=not clicked), implement K-armed bandit algorithms to maximize total reward (clicks).

## Brief Theory
**Multi-Armed Bandit Problem**: Choose actions (arms) to maximize cumulative reward. Balance **exploration** (try new arms) vs **exploitation** (choose best known arm).

- **Random**: Pure exploration, select randomly
- **ε-greedy**: ε% random (explore), (1-ε)% best so far (exploit)
- **UCB**: Optimistic upper bound: Q + √(2ln(t)/N)

## Implementation Explanation
1. Load `Advertising.csv` (10000 users × 10 ads)
2. **Random**: Select uniformly random ad each time
3. **ε-greedy**: ε=0.1 probability random, else argmax(Q)
4. **UCB**: Select arm maximizing Q(a) + √(2ln(N_total)/N(a))
5. Track selections, rewards, update Q-values
6. Plot histograms of ad selections

**Expected**: UCB performs best (optimistic exploration), followed by ε-greedy, random worst.

## Conclusion
UCB algorithm effectively balances exploration-exploitation, achieving highest cumulative reward. Suitable for online ad optimization where click probabilities unknown.

## References
1. Sutton & Barto, "Reinforcement Learning: An Introduction" Ch. 2
2. "Reinforcement Learning: An Introduction" - Multi-armed Bandits

## Setup & Run
```bash
pip install -r requirements.txt
jupyter notebook RL_EXP_2.ipynb
```

---


