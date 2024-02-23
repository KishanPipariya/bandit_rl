import numpy as np
class BanditProblem:
  def __init__(self, n_arms):
    self.n_arms = n_arms
    rng = np.random.default_rng(seed=42)
    vals = rng.integers(0, 3, (n_arms*2))
    vals = vals = vals.reshape((n_arms, 2))
    self.vals = vals

  def get_reward(self, chosen_arm):
    reward = np.random.normal(
        self.vals[chosen_arm][0],
        self.vals[chosen_arm][1])
    return reward

  def print_means_and_variances(self):
    print(self.vals)