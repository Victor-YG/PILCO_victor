import os
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as pp

import gym
import gpflow
from gpflow import set_trainable
import tensorflow as tf

from eval_utils import rollout, policy, rollout_with_expert
from pilco.models import PILCO
from pilco.controllers import CustomRbfController, LinearController, ExpertController, RbfController
from pilco.rewards import LinearReward, ExponentialReward, CombinedRewards

np.random.seed(1)
tf.compat.v1.enable_eager_execution()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", help="Path to saved expert controller.", default=None, required=False)
    parser.add_argument("--output", help="Path to save output results.", default="./swimmer", required=False)
    parser.add_argument("--timestep", help="Num of timestep in rollout", type=int, default=100, required=False)
    parser.add_argument("--iterations", help="Num of iteration to train model.", type=int, default=10, required=False)
    args = parser.parse_args()

    # local variables
    state_dim = 27
    control_dim = 8
    SUBS = 5
    maxiter = 80
    max_action = 1.0
    m_init = np.reshape(np.zeros(state_dim), (1, state_dim))  # initial state mean
    S_init = 0.005 * np.eye(state_dim)
    bf = 40

    # create output folder
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # create environment
    env = gym.make('Ant-v2').env

    # load expert controller
    expert_controller = None
    if args.expert is not None:
        print("Loding expert from {}".format(args.expert))
        expert_controller = ExpertController(os.path.abspath(args.expert))

    # initial rollout for setup dataset
    X, U, E, Y, _, __ = rollout_with_expert(env=env, pilco=None, expert=expert_controller, random=True, timesteps=args.timestep, SUBS=SUBS, render=True)
    X = X[:, 0 : state_dim]
    Y = Y[:, 0 : state_dim]

    # T = args.timestep
    # X = []
    # U = []
    # E = []
    # Y = []
    # while(T > 0):
    #     X_tmp, U_tmp, E_tmp, Y_tmp, _, __ = rollout_with_expert(env=env, pilco=None, expert=expert_controller, random=True, timesteps=args.timestep, SUBS=SUBS, render=True)
    #     X_tmp = X_tmp[:, 0 : state_dim]
    #     Y_tmp = Y_tmp[:, 0 : state_dim]
    #     X.append(X_tmp)
    #     U.append(U_tmp)
    #     E.append(E_tmp)
    #     Y.append(Y_tmp)
    #     T = T - X_tmp.shape[0]
    # X = np.vstack(X)
    # X = X[0 : args.timestep, 0 : state_dim]
    # U = np.vstack(U)
    # U = U[0 : args.timestep, :]
    # E = np.vstack(E)
    # E = E[0 : args.timestep, :]
    # Y = np.vstack(Y)
    # Y = Y[0 : args.timestep, 0 : state_dim]

    # create controller
    if expert_controller is not None:
        controller = CustomRbfController(X, E, state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    else:
        controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    # custom reward function
    # states:
    #   ...

    max_ang = 95 / 180 * np.pi
    reward_funcs = []

    # encourage speed to the right
    reward_funcs.append(LinearReward(state_dim, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # combined_reward = CombinedRewards(state_dim, reward_funcs, coefs=[1.0])

    # encourage staying above ground
    reward_funcs.append(ExponentialReward(state_dim,
        W=np.diag(np.array([10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + 1e-6),
        t=[0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    combined_reward = CombinedRewards(state_dim, reward_funcs, coefs=[1.0, 10.0])

    # create pilco
    pilco = PILCO((np.hstack((X, U)), Y), controller=controller, horizon=30, reward=combined_reward, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    # training
    rewards = []
    returns = []
    for i in range(args.iterations):
        print("---------------- Iteration {} ----------------".format(i + 1))
        start_time = time.time()

        # sample data
        X, U, E, Y, sampled_return, full_return = rollout_with_expert(env, pilco, expert_controller, timesteps=args.timestep, SUBS=SUBS, render=True)
        X = X[:, 0 : state_dim]
        Y = Y[:, 0 : state_dim]
        # if expert_controller is not None: pilco.controller.set_data((X, E))
        returns.append(full_return)

        # train model
        pilco.mgpr.set_data((np.hstack((X, U)), Y))
        pilco.optimize_models(maxiter=maxiter, restarts=1)
        reward = pilco.optimize_policy(maxiter=maxiter, restarts=1)
        rewards.append(reward.numpy()[0, 0])
        end_time = time.time()
        print("Iteration {} took {} seconds.".format(i + 1, end_time - start_time))

    # final rollout
    a, b, _, full_return = rollout(env, pilco, timesteps=200, SUBS=SUBS, render=True)
    returns.append(full_return)
    print("Final return = {}".format(full_return))

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