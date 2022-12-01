import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as pp

import gym
import gpflow
from gpflow import set_trainable
import tensorflow as tf

from eval_utils import rollout, policy
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController, ExpertController
from pilco.rewards import LinearReward, ExponentialReward, CombinedRewards

np.random.seed(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", help="Path to saved expert controller.", default=None, required=False)
    parser.add_argument("--output", help="Path to save output results.", default="./swimmer", required=False)
    parser.add_argument("--timestep", help="Num of timestep in rollout", type=int, default=100, required=False)
    parser.add_argument("--iterations", help="Num of iteration to train model.", type=int, default=10, required=False)
    args = parser.parse_args()

    # local variables
    state_dim = 8
    control_dim = 2
    SUBS = 5
    maxiter = 80
    max_action = 1.0
    m_init = np.reshape(np.zeros(state_dim), (1, state_dim))  # initial state mean
    S_init = 0.005 * np.eye(state_dim)
    T = 15
    bf = 40

    # create output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # create environment
    env = gym.make('Swimmer-v2').env

    # initial rollout for setup dataset
    X, Y, _, __ = rollout(env=env, pilco=None, random=True, timesteps=10, SUBS=SUBS, render=True)

    # create controller
    state_dim   = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    # custom reward function
    # states:
    #   0 - angle of the front tip
    #   1 - angle of the second rotor
    #   2 - angle of the third rotor
    #   3 - velocity of the tip along the x-axis
    #   4 - velocity of the tip along the y-axis
    #   5 - angular velocity of front tip
    #   6 - angular velocity of second rotor
    #   7 - angular velocity of third rotor
    max_ang = 95 / 180 * np.pi
    reward_funcs = []

    # encourage speed to the right
    reward_funcs.append(LinearReward(state_dim, [0, 0, 0, 10.0, 0, 0, 0, 0]))

    # dicourages second rotor from hitting the max angles
    reward_funcs.append(ExponentialReward(state_dim, W=np.diag(np.array([0, 10.0, 0, 0, 0, 0, 0, 0]) + 1e-6), t=[0,  max_ang, 0, 0, 0, 0, 0, 0]))
    reward_funcs.append(ExponentialReward(state_dim, W=np.diag(np.array([0, 10.0, 0, 0, 0, 0, 0, 0]) + 1e-6), t=[0, -max_ang, 0, 0, 0, 0, 0, 0]))

    # dicourages third rotor from hitting the max angles
    reward_funcs.append(ExponentialReward(state_dim, W=np.diag(np.array([0, 0, 10.0, 0, 0, 0, 0, 0]) + 1e-6), t=[0, 0,  max_ang, 0, 0, 0, 0, 0]))
    reward_funcs.append(ExponentialReward(state_dim, W=np.diag(np.array([0, 0, 10.0, 0, 0, 0, 0, 0]) + 1e-6), t=[0, 0, -max_ang, 0, 0, 0, 0, 0]))

    combined_reward = CombinedRewards(state_dim, reward_funcs, coefs=[1.0, -1.0, -1.0, -1.0, -1.0])

    # load expert controller
    expert_controller = None
    if args.expert is not None:
        expert_controller = ExpertController(os.path.abspath(args.expert))

    # create pilco
    pilco = PILCO((X, Y), controller=controller, expert=None, horizon=T, reward=combined_reward, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    ###########################
    # training without expert #
    ###########################
    rewards = []
    returns = []
    for i in range(args.iterations):
        print("---------------- Iteration {} ----------------".format(i + 1))

        # sample data
        X_new, Y_new, sampled_return, full_return = rollout(env, pilco, timesteps=args.timestep, SUBS=SUBS, render=True)
        returns.append(full_return)

        # update dataset
        X = np.vstack((X, X_new[:T,:]))
        Y = np.vstack((Y, Y_new[:T,:]))
        pilco.mgpr.set_data((X, Y))

        # train model
        pilco.optimize_models(maxiter=maxiter, restarts=2)
        reward = pilco.optimize_policy(maxiter=maxiter, restarts=2)
        rewards.append(reward.numpy().reshape(1, ))

    # create output folder
    if not os.path.exists(args.output):
           os.mkdir(args.output)

    # plot return and reward over time
    print("rewards = ")
    print(rewards)
    print("returns = ")
    print(returns)
    figure, axis = pp.subplots(2)
    axis[0].plot(range(args.iterations), rewards)
    axis[0].set_ylabel("rewards")
    axis[1].plot(range(args.iterations), returns)
    axis[1].set_xlabel("iterations")
    axis[1].set_ylabel("returns")
    pp.show()
    filepath = os.path.join(args.output, "results{}.png".format("_IL" if args.expert else ""))
    figure.savefig(filepath)


if __name__ == "__main__":
    main()