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


class DoublePendWrapper():
    '''
    Introduces a simple wrapper for the gym environment
    Reduces dimensions, avoids non-smooth parts of the state space that we can't model
    Uses a different number of timesteps for planning and testing
    Introduces priors
    '''

    def __init__(self):
        self.env = gym.make('InvertedDoublePendulum-v2').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        a1 = np.arctan2(s[1], s[3])
        a2 = np.arctan2(s[2], s[4])
        s_new = np.hstack([s[0], a1, a2, s[5:-3]])
        return s_new

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        if np.abs(ob[0])> 0.90 or np.abs(ob[-3]) > 0.15 or  np.abs(ob[-2]) > 0.15 or np.abs(ob[-1]) > 0.15:
            done = True
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob = self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()

    def close(self):
        return self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Path to save output results.", default="./inv_double_pendulum", required=False)
    parser.add_argument("--timestep", help="Num of timestep in rollout", type=int, default=100, required=False)
    parser.add_argument("--iterations", help="Num of iteration to train model.", type=int, default=5, required=False)
    args = parser.parse_args()

    # local variables
    SUBS = 1
    bf = 40
    maxiter = 10
    state_dim = 6
    control_dim = 1
    max_action = 1.0 # actions for these environments are discrete
    target = np.zeros(state_dim)
    weights = 5.0 * np.eye(state_dim)
    weights[0, 0] = 1.0
    weights[3, 3] = 1.0
    m_init = np.zeros(state_dim)[None, :]
    S_init = 0.005 * np.eye(state_dim)
    T = 40
    J = 5
    T_sim = 130
    restarts=True
    lens = []

    # create output folder
    if not os.path.exists(args.output):
           os.mkdir(args.output)

    # create environment
    env = DoublePendWrapper()

    # initial rollout for setup dataset
    X, Y, _, __ = rollout(env=env, pilco=None, random=True, timesteps=10, render=True)

    # create controller
    state_dim   = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller_RL = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)
    controller_IL = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=bf, max_action=max_action)

    # custom reward function
    reward_funcs = ExponentialReward(state_dim=state_dim, t=target, W=weights)

    # create pilco without expert
    pilco_RL = PILCO((X, Y), controller=controller_RL, expert=None, horizon=T, reward=reward_funcs, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco_RL.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    ###########################
    # training without expert #
    ###########################
    rewards_RL = []
    returns_RL = []
    for i in range(args.iterations):
        print("---------------- Iteration {} ----------------".format(i + 1))

        # sample data
        X_new, Y_new, sampled_return, full_return = rollout(env, pilco_RL, timesteps=args.timestep, verbose=True, SUBS=SUBS, render=True)
        returns_RL.append(full_return)

        # update dataset
        X = np.vstack((X, X_new[:T, :]))
        Y = np.vstack((Y, Y_new[:T, :]))
        pilco_RL.mgpr.set_data((X, Y))

        # train model
        pilco_RL.optimize_models(maxiter=maxiter, restarts=2)
        reward = pilco_RL.optimize_policy(maxiter=maxiter, restarts=2)
        rewards_RL.append(reward.numpy().reshape(1, ))


    ########################
    # training with expert #
    ########################
    # reset dataset
    X, Y, _, __ = rollout(env=env, pilco=None, random=True, timesteps=10, render=True)

    # create pilco with expert
    pilco_IL = PILCO((X, Y), controller=controller_IL, expert=pilco_RL.controller, horizon=T, reward=reward_funcs, m_init=m_init, S_init=S_init)

    # for numerical stability
    for model in pilco_IL.mgpr.models:
        model.likelihood.variance.assign(0.001)
        set_trainable(model.likelihood.variance, False)

    # training
    rewards_IL = []
    returns_IL = []
    for i in range(args.iterations):
        print("---------------- Iteration {} ----------------".format(i + 1))

        # sample data
        X_new, Y_new, sampled_return, full_return = rollout(env, pilco_IL, timesteps=args.timestep, verbose=True, SUBS=SUBS, render=True)
        returns_IL.append(full_return)

        # update dataset
        X = np.vstack((X, X_new[:T, :]))
        Y = np.vstack((Y, Y_new[:T, :]))
        pilco_IL.mgpr.set_data((X, Y))

        # train model
        pilco_IL.optimize_models(maxiter=maxiter, restarts=2)
        reward = pilco_IL.optimize_policy(maxiter=maxiter, restarts=2)
        rewards_IL.append(reward.numpy().reshape(1, ))


    ##########
    # result #
    ##########
    # plot return and reward over time
    figure, axis = pp.subplots(2)
    axis[0].set_ylabel("rewards")
    axis[0].plot(rewards_RL, label="w/o expert")
    axis[0].plot(rewards_IL, label="w/ expert")
    axis[0].legend()
    axis[1].set_xlabel("iterations")
    axis[1].set_ylabel("returns")
    axis[1].plot(returns_RL, label="w/o expert")
    axis[1].plot(returns_IL, label="w/ expert")
    axis[1].legend()
    filepath = os.path.join(args.output, "results.png")
    figure.savefig(filepath)
    pp.show()


if __name__=='__main__':
    main()