import sys
import random
import multiprocessing
import time
from time import perf_counter
from collections import deque

import numpy as np
from gym.spaces import Box

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

import matplotlib.pyplot as plt

import pickle

from src.envs.envs import BlackOilEnv
from sim_data_generation import StateActionTransition


def manh_dist(x, y, x1, y1):
    return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    # return abs(x - x1) + abs(y - y1)


# def get_done(state):
#     wells = 0
#     for i in state:
#         for j in i:
#             wells += j[-1]
#     return wells == 8


def get_experience_data(file_name: str):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    data_flatted = []
    for i in data:
        for j in i:
            j: StateActionTransition
            data_flatted.append((j.state, j.action, make_reward_number(j.reward), j.new_state, False))

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
        self.linear1 = nn.Linear(LAYER_SIZE + 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

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

        self.linear1 = nn.Linear(LAYER_SIZE, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        # print("BBB")
        try:
            state = torch.permute(state, (0, 3, 1, 2))
            # print('q')
            x = self.layer1(state)
            # print('a')
            # print(x.size())def make_reward_number(reward) -> float:
            #     if not isinstance(reward, float | np.float_):
            #         return reward[0]
            #     return reward
            x = self.layer2(x)
            # print('b')
            # print(x.size())
            x = self.layer3(x)
            # print('c')
            # print(x.size())

            x = F.relu(self.linear1(x))
            # print('d')
            x = F.relu(self.linear2(x))
            # print('e')
            x = F.relu(self.linear3(x))
            # print('f')
            # print("XXX", x, x.size())
            # x = self.sigmoid(x)
            x = F.sigmoid(x)
            # print('f')
            # print(x)

            return x
        except Exception as exc:
            print(exc)

        return 0, 0


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
        self.actor.share_memory()
        self.critic.share_memory()
        self.critic_target.share_memory()
        self.actor_target.share_memory()

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
        # print(1)
        with torch.no_grad():
            # print(2)
            state = torch.from_numpy(state).float().unsqueeze(0)
            # print(3)
            # print(state)
            action = self.actor.forward(state)
            # print(4)
            # action = np.array(action[0], action[1])
            action = action.detach().numpy()[0]
        return action

    def update(self, target_update):
        states, actions, rewards, next_states, done = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done)

        # print(states.size())
        # print(actions.size())
        # Critic loss
        Qvals = self.critic.forward(states, actions)
        with torch.no_grad():
            next_actions = self.actor_target.forward(next_states)
            next_Q = self.critic_target.forward(next_states, next_actions)
            Qprime = rewards + self.gamma * next_Q * (1 - done)
        critic_loss = self.critic_criterion(Qvals, Qprime)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # print(Qvals)
        # print(Qvals.size())
        print(f"Qvals {Qvals.mean()}, next_Q {next_Q.mean()}, Qprime {Qprime.mean()}, rewards {rewards.mean()}, critic_loss {critic_loss}")
        q_val.append((Qvals.mean(), next_Q.mean(), Qprime.mean()))
        loss_val.append(critic_loss)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        if target_update:

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


class Environment(BlackOilEnv):
    def __init__(self, w=80, h=40, wells=8, days=30):
        super().__init__(w=w, h=h, wells=wells, days=days)
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([self.w, self.h]), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=0.0, shape=(self.h, self.w, 8), dtype=np.float32)


def check_well(wells, x, y, intersection_radius=2):
    if 0 <= x < W and 0 <= y < H:
        for w in wells:
            if abs(w[0] - x) < intersection_radius or abs(w[1] - y) < intersection_radius:
                return False
        return True
    return False


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


def help_action(wells, action):
    if check_well(wells, *action):
        return action
    min_dist = 10000
    pos = []
    for x in range(W):
        for y in range(H):
            if check_well(wells, x, y):
                if manh_dist(x, y, *action) < min_dist:
                    min_dist = manh_dist(x, y, *action)
                    pos = [(x, y)]
                elif manh_dist(x, y, *action) == min_dist:
                    pos.append((x, y))
    return random.choice(pos)


q_val = []
loss_val = []


# NUM_PROCESSES = multiprocessing.cpu_count()
NUM_PROCESSES = 1
H = 40  # 40
W = 80  # 80
WELLS = 8  # 8
DAYS = 30  # 30
LAYER_SIZE = 3200  # 3200 | 384
MAX_EPISODES = 7  # 500
BATCH_SIZE = 128  # 128
MEMORY_SIZE = 50_000  # 50_000
TARGET_UPDATE_TIME = 10
NOISE_RANGE = 5  # 20
TRIES_TO_BUILT_WELL = 50
DATA = ['saved_results1.pkl', 'saved_results2.pkl', 'saved_results3.pkl', 'saved_results5.pkl']

# TODO : логировать всевозможные параметры, чтобы было легче дебажить

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


# def step_over_episode(episode_number, rewards):
def step_over_episode(args):
    agent, episode_number, rewards, steps = args
    # print(episode_number, rewards)

    env = Environment(w=W, h=H, wells=WELLS, days=DAYS)
    # noise = OUNoise(env.action_space)
    # noise.reset()

    state = env.reset()
    episode_reward = 0
    time_step, time_update = [], []
    time_begin = perf_counter()
    wells = []

    for step in range(WELLS):

        action = agent.get_action(state)

        # TODO : рандомный шум - насколько большой ?
        action = np.clip(np.tanh(torch.randn(2)) / NOISE_RANGE + torch.tensor(action), 0, 1).tolist()
        # action = noise.get_action(action).tolist()

        action = get_norm_coord(action)
        action = help_action(wells, action)

        time_per_step = perf_counter()
        new_state, reward, done = env.step(action)
        time_per_step = perf_counter() - time_per_step

        wells.append(action)

        reward = make_reward_number(reward)
        # agent.memory.push(state, action, reward * beta, new_state, done)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > BATCH_SIZE:
            # print(episode_number, step)
            time_per_update = perf_counter()
            agent.update(steps % TARGET_UPDATE_TIME == 0)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

            time_per_update = perf_counter()
            agent.update(steps % TARGET_UPDATE_TIME == 0)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

            time_per_update = perf_counter()
            agent.update(steps % TARGET_UPDATE_TIME == 0)
            time_per_update = perf_counter() - time_per_update
            time_update.append(time_per_update)

        print(
            f"{step + 1}: action: {action}, TIME per step: {time_per_step}, TIME per update: {np.mean(time_update[-3:])}")

        steps += 1
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
    env = Environment(w=W, h=H, wells=WELLS, days=DAYS)
    agent = DDPGagent(env, max_memory_size=MEMORY_SIZE)

    steps = 0
    rewards = []
    avg_rewards = []
    number_wells = {i + 1: 0 for i in range(WELLS)}
    step_time = []
    update_time = []
    episode_time = []

    # Считаем максимальную и среднюю нагруду по рандомным данным - равна 2.7 : 0.5
    tt = get_experience_data(DATA[0]) + get_experience_data(DATA[1]) + get_experience_data(DATA[2]) + get_experience_data(DATA[3])
    max_r = np.mean([x[2] for x in tt])
    print(max_r)

    # Считаем максимальную и среднюю стоимость вышки по рандомным данным - равна 2.45 : 4.06
    max_b = np.mean([c[6] for x in tt for y in x[0] for c in y])
    print(max_b)

    if H == 40 and W == 80 and WELLS == 8:
        # Загружаем данные в память
        for file_name in DATA:
            temp = get_experience_data(file_name)
            for sample in temp:
                agent.memory.push(*sample)
        print("DATA LOADED")

    for episode in range(MAX_EPISODES):
        actions = [(agent, episode + i / 10, rewards[-10:], steps) for i in range(1, NUM_PROCESSES + 1)]

        # with multiprocessing.Pool(NUM_PROCESSES) as pool:
        #     episode_reward, wells_built, time_step, time_update, time_begin = pool.map(step_over_episode, actions)
        # try:
        episode_reward, wells_built, time_step, time_update, time_begin = step_over_episode(actions[0])
        # except Exception as exc:
        #     print(exc)

        # if len(agent.memory) > BATCH_SIZE:
        #     time_per_update = perf_counter()
        #     agent.update(BATCH_SIZE)
        #     time_per_update = perf_counter() - time_per_update
        #     time_update.append(time_per_update)
        #
        #     time_per_update = perf_counter()
        #     agent.update(BATCH_SIZE)
        #     time_per_update = perf_counter() - time_per_update
        #     time_update.append(time_per_update)
        #
        #     time_per_update = perf_counter()
        #     agent.update(BATCH_SIZE)
        #     time_per_update = perf_counter() - time_per_update
        #     time_update.append(time_per_update)

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
    plt.grid()
    plt.show()
    plt.close()

    # /// diagram
    plt.bar(number_wells.keys(), number_wells.values())
    plt.plot()
    plt.xlabel('Wells')
    plt.ylabel('Episode')
    plt.grid()
    plt.show()
    plt.close()

    plt.plot(episode_time, label='episode')
    plt.plot(step_time, label='step')
    plt.plot(update_time, label='update')
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    plt.plot([x[0].detach() for x in q_val], label='Qval')
    plt.plot([x[1].detach() for x in q_val], label='next_Q')
    plt.plot([x[2].detach() for x in q_val], label='Qprime')
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    plt.plot(list(map(torch.detach, loss_val)), label='critic_loss')
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
