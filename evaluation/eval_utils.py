import numpy as np
import tensorflow as tf
from gpflow import config
from gym import make
float_type = config.default_float()


def rollout(env, pilco, timesteps, verbose=False, random=False, SUBS=1, render=False):
        X = []
        Y = []
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            if done: break
        print("Return so far: ", ep_return_full)
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def rollout_with_expert(env, pilco, expert, timesteps, verbose=False, random=False, SUBS=1, render=False):
        X = []
        Y = []
        U = []
        E = []
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0

        for timestep in range(timesteps):
            if render: env.render()

            u = policy(env, pilco, x, random)
            if expert is not None: u_e = expert.compute_action(x)[0]

            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                ep_return_full += r
                if done: break
                if render: env.render()

            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)

            X.append(x)
            U.append(u)
            Y.append(x_new - x)

            if expert is not None:
                E.append(u_e)
            else:
                E.append(np.zeros_like(u))

            ep_return_sampled += r
            x = x_new
            if done: break

        print("Return so far: ", ep_return_full)
        return np.stack(X), np.stack(U), np.stack(E), np.stack(Y), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):
    if random or pilco is None:
        return env.action_space.sample()
    else:
        return tf.squeeze(pilco.compute_action(x)[0])

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()
