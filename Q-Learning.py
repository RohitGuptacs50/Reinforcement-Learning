import gym
import numpy as np

domain = gym.make("MountainCar-v0")
print(domain.action_space.n)    # 3 actions can be taken, 0: left, 1: still, 2: right
print(domain.reset())

domain.reset()    # reset the domain, returns an initial observation

done = False
while not done:
    action = 2    # always go right
    domain.step(action)
    domain.render()


state = domain.reset()

done = False
while not done:
    action = 2
    new_state, reward, done, _ = domain.step(action)     # Each step is worth -1 reward and flag is worth 0 reward
    print(reward, new_state)                                # state is array of 2 values of position and velocity along x/horizontal axis 
                                                            # so, if car is moving left, velocity is negative

print(domain.observation_space.high)            # highest possible state values(position and velocity) of the domain
print(domain.observation_space.low)             # lowest possible state values(position and velocity) of the domain 

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (domain.observation_space.high - domain.observation_space.low) / DISCRETE_OS_SIZE   # to contain the size of the q-table

print(discrete_os_win_size)

q_table = np.random.uniform(low = -2, high=0, size=(DISCRETE_OS_SIZE + [domain.action_space.n]))    # values for each action per each state
print(q_table)                                                          # when greedy, will go with the action that has highest q value for a perticular state