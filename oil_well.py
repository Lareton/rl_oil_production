# Genetic Algorithm for oil wells

import random
import tqdm
import numpy as np
from src.envs.envs import BlackOilEnv
from src.render.render import render
from src.utils.map_generation.generate_linked_graph import generate_graph
import matplotlib.pyplot as plt

def individual(state):
        chrom = []
        for _ in range(8):   # Individual, 'chromosome'
            action = (random.randint(0, state.shape[1] - 1), random.randint(0, state.shape[0] - 1))
            x, y = action

            while state[y][x][4] < 1e-6: # Checking the oil_amount
                action = (random.randint(0, state.shape[1] - 1), random.randint(0, state.shape[0] - 1))
                x, y = action
            chrom.append(action)
        return chrom

def initialPopulation(state):
        population = []
        for _ in range(10):
            population.append(individual(state))
        return population

def fitness_function(population, env):
        best = []
        for pop in tqdm.tqdm(population):
            sum_reward_pop = 0
            env.reset()
            for chr in pop:
                state, reward, done = env.step(chr)
                try:
                    sum_reward_pop += reward
                except Exception as exc:
                    sum_reward_pop += reward[0]
            best.append((pop, sum_reward_pop))
        best.sort(key=lambda x: -x[1])
        print(best)
        popul = [i[0] for i in best]
        return popul

def crossover(parents):
        children = []
        for _ in range(10):
            n = random.sample(parents, 2)
            genes1 = random.sample(n[0], 4)
            genes2 = random.sample(n[1], 4)
            child = random.sample(genes1 + genes2, k=8)
            children.append(child)

        return children

def mutation(popul, state):
    for i in range(10):
        n = random.randint(0, 7)
        action = (random.randint(0, state.shape[1] - 1), random.randint(0, state.shape[0] - 1))
        x, y = action

        while state[y][x][4] < 1e-6:
            action = (random.randint(0, state.shape[1] - 1), random.randint(0, state.shape[0] - 1))
            x, y = action
        
        popul[i][n] = (x, y)
    return popul


# Generating the ENV
if __name__ == "__main__":
    env = BlackOilEnv()
    np.random.seed(1)
    pre_state = env.reset()

    """
    popul = initialPopulation(state)
    best_popul = fitness_function(popul, env)
    popul = crossover(best_popul[:4])
    """

    best_generation = initialPopulation(pre_state)
    for _ in range(10):
        best_generation = crossover(best_generation[:4])
        best_generation = mutation(best_generation, pre_state)
        best_generation = fitness_function(best_generation, env)
        print(best_generation[0])
    pre_state = env.reset()
    for action in best_generation[0]:
        state, reward, done = env.step(action)
    render(pre_state[:, :, 4], state,  best_generation[0], env)


''' pre_state = env.reset()
      for action in best_generation[0]:
        state, reward, done = env.step(action)
        print(reward)
    env.render(settings)'''
