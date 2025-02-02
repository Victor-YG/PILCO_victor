import os
import pickle
import numpy as np
import matplotlib.pyplot as pp
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
import gpflow
from gpflow import set_trainable
# from tensorflow import logging
np.random.seed(0)

from utils import rollout, policy

env = gym.make('InvertedPendulum-v2')

# Initial random rollouts to generate a dataset
X,Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=40, render=True)
for i in range(1,5):
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=40, render=True)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
# controller = LinearController(state_dim=state_dim, control_dim=control_dim)

if os.path.exists("inv_pend_expert_controller.pkl"):
    with open("inv_pend_expert_controller.pkl", "rb") as f:
        expert_controller = pickle.load(f)
    pilco = PILCO((X, Y), controller=controller, horizon=40, expert=expert_controller)
else:
    pilco = PILCO((X, Y), controller=controller, horizon=40)

n_iterations = 5
rewards = []
returns = []
for i in range(n_iterations):
    print("---------------- Iteration {} ----------------".format(i + 1))
    pilco.optimize_models()
    reward = pilco.optimize_policy()
    rewards.append(reward.numpy().reshape(1,))
    import pdb; pdb.set_trace()
    X_new, Y_new, sampled_return, full_return = rollout(env=env, pilco=pilco, timesteps=100, render=True)
    # Update dataset
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X, Y))
    returns.append(full_return)

# plot return over time
# pp.plot(range(n_iterations), returns)
pp.plot(range(n_iterations), rewards)
pp.xlabel("iterations")
pp.ylabel("reward")
pp.show()

save_trained_model = True
# save trained model
if save_trained_model:
    with open("inv_pend_expert_controller.pkl", "wb") as f:
        pickle.dump(pilco.controller, f)