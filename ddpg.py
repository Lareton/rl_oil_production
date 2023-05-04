import sys
import random
import multiprocessing
from time import perf_counter
from collections import deque

import numpy as np
from gym.spaces import Box

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

import matplotlib.pyplot as plt

import pickle

from src.envs.envs import BlackOilEnv
from sim_data_generation import StateActionTransition


def get_experience_data(file_name: str):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    data_flatted = []
    for i in data:
        for j in i:
            j: StateActionTransition
            data_flatted.append((j.state, j.action, make_reward_number(j.reward), j.new_state))

    return data_flatted


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 2
        self.low = 0.
        self.high = 1.
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

    def push(self, state, action, reward, next_state):
        experience = (state, action, np.array([reward]), next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            nn.Flatten(1)
        )
        self.linear1 = nn.Linear(3202, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        # action = self.flatten(action)
        # print("AAA")
        # print(state.size())
        state = torch.permute(state, (0, 3, 1, 2))
        # print(state.size())
        x = self.layer1(state)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())

        # print(action.size())
        x = torch.cat([x, action], 1)
        # print(x.size())
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            nn.Flatten(1)
        )

        self.linear1 = nn.Linear(3200, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        # print("BBB")
        state = torch.permute(state, (0, 3, 1, 2))
        x = self.layer1(state)
        # print(x.size())def make_reward_number(reward) -> float:
        #     if not isinstance(reward, float | np.float_):
        #         return reward[0]
        #     return reward
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # print("XXX", x, x.size())
        # x = self.sigmoid(x)
        x = F.sigmoid(x)
        # print(x)

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

    def update(self, BATCH_SIZE):
        states, actions, rewards, next_states = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # print(states.size())
        # print(actions.size())
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


def get_norm_coord(action) -> tuple:
    action[0] *= (W - 1)
    action[1] *= (H - 1)
    if not (0 <= action[0] < W or 0 <= action[1] < H):
        raise ValueError("NaN here")
    return tuple(map(round, action))


def make_reward_number(reward) -> float:
    if not isinstance(reward, float | np.float_):
        return reward[0]
    return reward


env: Environment
agent: DDPGagent

NUM_PROCESSES = multiprocessing.cpu_count()
H = 40  # 40
W = 80  # 80
WELLS = 4  # 8
DAYS = 3  # 30
MAX_EPISODES = 2  # 500
BATCH_SIZE = 128  # 128
MEMORY_SIZE = 50_000  # 50_000
NOISE_RANGE = 20  # 20
DATA = ['saved_results1.pkl', 'saved_results2.pkl', 'saved_results3.pkl', 'saved_results5.pkl']


"""
    Как запускать тестирование:
        1) Проверить соответствие WELLS к (W, H)
        2) Осознать правильность DAYS
        3) Осознать правильность MAX_EPISODES
        4) Вспомнить про BATCH_SIZE и NOISE_RANGE
        5) Надеяться на лучшее и запустить
    
    Как тестить:
        1) Загружаем модель
        2) Загружаем карту
        3) Чисто тестим: без шума и обучения
"""


def step_over_episode(episode_number, rewards):
    global env, agent

    state = env.reset()
    episode_reward = 0
    time_step, time_update = [], []
    time_begin = perf_counter()

    for step in range(WELLS):
        action = agent.get_action(state)

        # TODO : рандомный шум - насколько большой ?
        action = np.clip(np.tanh(torch.randn(2)) / NOISE_RANGE + torch.tensor(action), 0, 1).tolist()
        action = get_norm_coord(action)

        time_per_step = perf_counter()
        new_state, reward, _ = env.step(action)
        time_per_step = perf_counter() - time_per_step

        reward = make_reward_number(reward)
        agent.memory.push(state, action, reward, new_state)

        if len(agent.memory) > BATCH_SIZE:
            time_per_update = perf_counter()
            agent.update(BATCH_SIZE)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

            time_per_update = perf_counter()
            agent.update(BATCH_SIZE)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

            time_per_update = perf_counter()
            agent.update(BATCH_SIZE)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

        print(
            f"{step + 1}: action: {action}, TIME per step: {time_per_step}, TIME per update: {np.mean(time_update[-3:])}")

        state = new_state
        episode_reward += reward

        time_step.append(time_per_step)

    time_begin = perf_counter() - time_begin

    wells_built = 0
    for x in range(H):
        for y in range(W):
            if state[x][y][-1]:
                wells_built += 1

    sys.stdout.write(
        "episode: {}, reward: {}, average _reward: {}, wells_number: {}, time: {}\n".format(episode_number,
                                                                                            np.round(episode_reward,
                                                                                                     decimals=2),
                                                                                            np.mean(rewards),
                                                                                            wells_built,
                                                                                            time_begin))

    return episode_reward, wells_built, time_step, time_update, time_begin


def main():
    global env, agent

    env = Environment(w=W, h=H, wells=WELLS, days=DAYS)
    agent = DDPGagent(env, max_memory_size=MEMORY_SIZE)

    rewards = []
    avg_rewards = []
    number_wells = {i + 1: 0 for i in range(WELLS)}
    step_time = []
    update_time = []
    episode_time = []

    # Считаем максимальную нагруду по рандомным данным - равна 2.7
    # tt = get_experience_data(DATA[0])[1]
    # tt = max(tt, get_experience_data(DATA[1])[1])
    # tt = max(tt, get_experience_data(DATA[2])[1])
    # tt = max(tt, get_experience_data(DATA[3])[1])
    # print(tt)

    for file_name in DATA:
        temp = get_experience_data(file_name)
        for sample in temp:
            agent.memory.push(*sample)
    print("DATA LOADED")

    for episode in range(MAX_EPISODES):
        actions = [episode + i / 10 for i in range(1, NUM_PROCESSES + 1)]

        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            episode_reward, wells_built, time_step, time_update, time_begin = pool.map(step_over_episode, actions)

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        number_wells[wells_built] += 1
        step_time.append(np.mean(time_step))
        update_time.append(np.mean(time_update))
        episode_time.append(time_begin)

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    plt.close()

    # /// diagram
    plt.bar(number_wells.keys(), number_wells.values())
    plt.plot()
    plt.xlabel('Wells')
    plt.ylabel('Episode')
    plt.show()
    plt.close()

    plt.plot(episode_time, label='episode')
    plt.plot(step_time, label='step')
    plt.plot(update_time, label='update')
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
