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
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward


np.random.seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", help="Path to saved expert controller.", default=None, required=False)
    parser.add_argument("--output", help="Path to save output results.", default="./inverted_pendulum", required=False)
    parser.add_argument("--timestep", help="Num of timestep in rollout", type=int, default=100, required=False)
    parser.add_argument("--horizon", help="Planning horizon.", default=10, required=False)
    parser.add_argument("--iterations", help="Num of iteration to train model.", type=int, default=5, required=False)
    parser.add_argument("-save_expert", help="Whether to save the trained controller.", action="store_true", required=False)
    args = parser.parse_args()

    # create environment
    env = gym.make('InvertedPendulum-v2')

    # initial rollout for setup dataset
    X, Y, _, __ = rollout(env=env, pilco=None, random=True, timesteps=10, render=True)

    # create controller
    state_dim   = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller  = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)

    # load expert controller
    expert_controller = None
    if args.expert is not None and os.path.exists(args.expert):
        with open(args.expert, "rb") as f:
            expert_controller = pickle.load(f)

    # create pilco
    pilco = PILCO((X, Y), controller=controller, horizon=40, expert=expert_controller)

    # training
    rewards = []
    returns = []
    for i in range(args.iterations):
        print("---------------- Iteration {} ----------------".format(i + 1))

        # sample data
        X_new, Y_new, sampled_return, full_return = rollout(env=env, pilco=pilco, timesteps=args.timestep, render=True)
        returns.append(full_return)

        # update dataset
        X = np.vstack((X, X_new))
        Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))

        # train model
        pilco.optimize_models()
        reward = pilco.optimize_policy()
        rewards.append(reward.numpy().reshape(1, ))

    # create output folder
    if not os.path.exists(args.output):
           os.mkdir(args.output)

    # plot return and reward over time
    figure, axis = pp.subplots(2)
    axis[0].plot(range(args.iterations), rewards)
    axis[0].set_ylabel("rewards")
    axis[1].plot(range(args.iterations), returns)
    axis[1].set_xlabel("iterations")
    axis[1].set_ylabel("returns")
    pp.show()
    filepath = os.path.join(args.output, "results{}.png".format("_IL" if args.expert else "_RL"))
    figure.savefig(filepath)

    # save trained controller
    if args.save_expert:
        with open(os.path.join(args.output, "expert_controller.pkl"), "wb") as f:
            pickle.dump(pilco.controller, f)


if __name__ == "__main__":
    main()