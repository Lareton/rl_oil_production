from src.envs.envs import BlackOilEnv, BaseBlackOilEnv
from src.render.render import RenderSettings

import random
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import Tuple
from time import time
import pickle

SEED = 42
NUM_PROCESSES = 16
NUM__TRANSITION_SAMPLES = 1000
NUM_ACTIONS = 8

np.random.seed(SEED)
random.seed(SEED)


@dataclass
class StateActionTransition:
    """Класс для хранения state-action перехода"""
    state: np.ndarray
    new_state: np.ndarray
    action: Tuple[int, int]
    reward: np.float64


def gen_random_action(env: BaseBlackOilEnv):
    return random.randint(0, env.w-1), random.randint(0, env.h-1)


def gen_random_actions(env: BaseBlackOilEnv, count_actions):
    return [gen_random_action(env) for _ in range(count_actions)]


def sim_tick(actions):
    print(f"start sim: {actions}")
    env = BlackOilEnv()
    env.reset()
    transitions = []

    ts = time()
    for action_num in range(NUM_ACTIONS):
        print(action_num)
        action = actions[action_num]

        state = env.observation
        new_state, reward, done = env.step(action)
        transition = StateActionTransition(state, new_state, action, reward)
        transitions.append(transition)
    te = time()

    print("spent time: ", te - ts)
    return transitions


def main():
    all_results = []

    for step_num in range(NUM__TRANSITION_SAMPLES // (NUM_ACTIONS * NUM_PROCESSES)):
        actions = [gen_random_actions(BlackOilEnv(), NUM_PROCESSES)
                   for _ in range(NUM_PROCESSES)]

        with multiprocessing.Pool(NUM_PROCESSES) as p:
            transitions = p.map(sim_tick, actions)

        all_results.extend(transitions)

        if step_num % 10 == 0:
            with open("saved_results.pkl", "wb") as f:
                pickle.dump(all_results, f)


if __name__ == "__main__":
    main()
