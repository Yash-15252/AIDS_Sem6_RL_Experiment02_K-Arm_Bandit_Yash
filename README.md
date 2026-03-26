# AIDS_Sem8_RL_Experiment02_Bandit

## ***YASH KHAMKAR - 221A030***
## ***K-armed Bandit Problem: 10-Arm Ad Selection***

---

## Aim
Solve **10-armed Multi-armed Bandit (MAB)** problem for ad selection using **Epsilon-Greedy** algorithm. Compare random, epsilon-greedy, and optimal performance.

## Problem Statement
```
• 10 different ads (arms/actions) on webpage
• Each column = 1 ad, rows = user impressions  
• Reward: 1 = click (positive), 0 = no-click
• Goal: Maximize cumulative clicks over N steps (explore vs exploit)
```

**Input**: Click probability matrix [users × ads]

## Brief Theory
**Multi-Armed Bandit (MAB)** Framework:
```
Multi-Armed Bandit Problem: k slot machines (arms) with unknown reward distributions μ₁, μ₂, ..., μₖ
Agent must balance Exploration (try different arms) vs Exploitation (pull highest estimated reward arm)
```

**Epsilon-Greedy Algorithm** (ε-Greedy):
```
With probability ε:  Explore (random arm)
With probability 1-ε: Exploit (argmax Q̂_t(a))
Q̂_t+1(a) = Q̂_t(a) + (1/(t+1))(r_{t+1} - Q̂_t(a))
```

**Performance Metrics**:
- **Average Reward**: ∑r_t / T
- **Optimal Action %**: Steps where π*(s)=argmax μ_a was selected

## Implementation Explanation
`RL_EXP_2.ipynb` implements:

```
1. Dataset: ad-click matrix (1=click, 0=no-click)
2. Random Policy: Uniform random arm selection baseline
3. ε-Greedy: ε=0.1, decaying α, multi-run averaging
4. Q-Learning Updates: Incremental average Q(a) estimates
5. Metrics: Average reward, % optimal action, regret
6. Visualization: Learning curves (random vs ε-greedy vs optimal)
```

**Key Algorithm** (ε-Greedy):
```
Initialize Q(a)=0 ∀a∈{1..10}
For each step t=1 to T:
    ε = 0.1
    a = ε-random else argmax Q(a)
    r ~ Bernoulli(p_a)  # True click prob
    Q(a) ← Q(a) + (1/t)(r - Q(a))
```

## Results
```
Random Policy: ~baseline CTR (average p_a ≈ 0.1-0.3)
ε-Greedy (ε=0.1): Converges to optimal arm, %Optimal → 90%+
Optimal Policy: Upper bound (100% best arm selection)

Regret: ε-Greedy << Random (exploit learned best arm)
```

**Key Observation**: ε-Greedy discovers optimal ad while random never improves.

## Sample Output
```
Average Reward vs Steps:
├── Random: Flat ~0.2 CTR
├── ε=0.1: ↑ Converges ~0.45 CTR  
└── Optimal: Theoretical max ~0.5 CTR

% Optimal Action:
├── Random: ~10% 
├── ε=0.1: ↑ 90% after 5000 steps ✓
```

## Conclusion
 **ε-Greedy Success**: Balances explore/exploit, discovers optimal ad  
 **Regret Minimization**: Much better than random after convergence  
 **Practical**: Directly applicable to A/B testing, ad rotation  
 **Scalability**: Extends to Thompson Sampling, UCB variants  

**Performance**: ε-Greedy achieves 90% optimal arm selection, 2× random CTR.

## References
1. Sutton & Barto, "RL: An Introduction" (Ch. 2: Multi-armed Bandits)
2. K-armed Bandit: ε-Greedy Algorithm
3. AIDS Sem8 RL Course (Experiment 2)

## Setup & Run
```bash
cd AIDS_Sem8_RL_Experiment02_Bandit
pip install -r requirements.txt
jupyter notebook RL_EXP_2.ipynb
```

**Requirements**:
```
numpy
matplotlib
pandas
```

---

