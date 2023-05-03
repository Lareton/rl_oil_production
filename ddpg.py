import numpy as np
import gym
from gym.spaces import Box
from collections import deque
import random

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

from src.envs.envs import BlackOilEnv

import sys
import pandas as pd
import matplotlib.pyplot as plt


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


def norm_action(self, action):  # self = env
    act_k = (self.action_space.high - self.action_space.low) / 2.
    act_b = (self.action_space.high + self.action_space.low) / 2.
    return act_k * action + act_b


def reverse_action(self, action):  # self = env
    act_k_inv = 2. / (self.action_space.high - self.action_space.low)
    act_b = (self.action_space.high + self.action_space.low) / 2.
    return act_k_inv * (action - act_b)


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        # TODO : Нормально сворачивать - какие параметры и куда ставить
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, (2, 2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv3d(16, 32, (3, 3, 3)),
        #     nn.MaxPool2d(kernel_size=(2, 2)),
        #     nn.ReLU(),
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Conv3d(32, 64, (4, 4, 4)),
        #     nn.MaxPool2d(kernel_size=(2, 2)),
        #     nn.ReLU(),

            # TODO : подобрать параметры для Flatten
            nn.Flatten(0),
            nn.Sigmoid()
        )
        # TODO : Каковы параметры слоев
        self.linear1 = nn.Linear(1728, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = self.layer1(state)
        print(x.size())
        # x = self.layer2(x)
        # print(x.size())
        # x = self.layer3(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class DDPGagent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):
        # Params
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        # print(env.observation_space.shape[0], env.action_space.shape[0])
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        # print(state)
        action = self.actor.forward(state)
        # action = np.array(action[0], action[1])
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


class Environment(BlackOilEnv):
    def __init__(self, w=80, h=40, wells=8, days=30):
        super().__init__(w=w, h=h, wells=wells, days=days)
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([self.w, self.h]), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=0.0, shape=(self.h, self.w, 8), dtype=np.float32)


def get_norm_coord(action):
    return list(map(lambda x: 0 if x == 0 else round(np.log((1 / (x + 1e-8)) - 1)), action))


def main():
    env = Environment(w=10, h=10, wells=4, days=5)

    agent = DDPGagent(env)
    noise = OUNoise(env.action_space)
    batch_size = 128
    rewards = []
    avg_rewards = []
    print(agent)

    for episode in range(50):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        done = False

        for step in range(4):
            action = agent.get_action(state)
            action = noise.get_action(action, step)
            # print(action)
            action = get_norm_coord(action)
            print(action)
            # print(state.shape)
            new_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > batch_size:
                agent.update(batch_size)

            state = new_state
            episode_reward += reward

            if done:
                sys.stdout.write(
                    "episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                             np.round(episode_reward, decimals=2),
                                                                             np.mean(rewards[-10:])))
                # env.render()
                break

        assert done is True

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()
