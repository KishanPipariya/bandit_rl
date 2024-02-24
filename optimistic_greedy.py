from bandit_prob import BanditProblem
import numpy as np
alpha = 0.1

n_arms=3

bandit = BanditProblem(n_arms)

total_reward = 0
q_vals = np.zeros(n_arms)
sum_rewards = np.zeros(n_arms)
n_counts = np.ones(n_arms)

q_vals = np.full(n_arms, 10, dtype=np.float32)
for i in range(1000):
  chosen_arm = np.argmax(q_vals)
  reward = bandit.get_reward(chosen_arm)
  total_reward += reward
  q_vals[chosen_arm] = q_vals[chosen_arm] + (reward - q_vals[chosen_arm])*alpha
print(q_vals)

bandit.print_means_and_variances()