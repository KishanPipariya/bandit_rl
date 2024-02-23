from bandit_prob import BanditProblem
import numpy as np

n_arms = 4
bandit = BanditProblem(n_arms)

total_reward = 0
q_vals = np.zeros(n_arms)
sum_rewards = np.zeros(n_arms)
n_counts = np.ones(n_arms)


epsilon = 0.3
for i in range(10000):
  chosen_arm = 0
  if bandit.rng.random() > epsilon:
    chosen_arm = np.argmax(q_vals)
  else:
    chosen_arm = bandit.rng.integers(0, n_arms)
  reward = bandit.get_reward(chosen_arm)
  total_reward += reward
  q_vals[chosen_arm] = q_vals[chosen_arm] + (reward - q_vals[chosen_arm])/n_counts[chosen_arm]
  n_counts[chosen_arm] += 1
print(q_vals)

bandit.print_means_and_variances()

