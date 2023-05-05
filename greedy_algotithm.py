from src.utils.map_generation.generate_linked_graph import generate_graph
from src.render.render import RenderSettings
from src.envs.envs import BlackOilEnv
import numpy as np
import random
import numpy as np

SEED = 1
np.random.seed(SEED)
#random.seed(SEED)
ans=[]

def test():
    #settings = RenderSettings()
    #settings.display_ground = True
    #settings.display_wells = True

    env = BlackOilEnv()
    state = env.reset()
    #env.render(settings)
    all_reward = 0

    kof = 3
    #max_S=[[0, 0], -10]
    m = []
    for _ in range(8):
        max_S=[[0, 0], -10]
        for i in range(80):
            for j in range(40):
                p = state[j][i][4] * kof - state[j][i][6]
                if max_S[1] <= p and [i, j] not in m:
                    rep = 0
                    for i1 in range(len(m)):
                        if abs(m[i1][0] - i) + abs(m[i1][1] - j)>=3:
                            rep +=1
                    if rep == len(m):
                        max_S[0][0] = i
                        max_S[0][1] = j
                        max_S[1] = p

        action = max_S[0]
        state, reward, done = env.step(action)
        #env.render(settings)
        all_reward+=reward
        m.append(max_S[0])
        #print(reward, max_S[1], reward -max_S[1])
    ans.append(all_reward)
    return all_reward

for i in range(1, 21):
    SEED = i
    print(test())
    print(i, "СР.Знач", sum(ans)/len(ans))
print(sum(ans)/20)
