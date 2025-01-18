import gym
import numpy as np
env = gym.make("LunarLander-v2", render_mode="human")

observation = env.reset(seed=42)

for _ in range(1000):
   print("observations are: ", observation)

   action = 0
   observation, reward, terminated, _ = env.step(action)

   if terminated:
      observation = env.reset()
env.close()