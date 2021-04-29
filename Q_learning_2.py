# objective: cart to the flag.
# for now, let's just move randomly:


import gym
import numpy as np
import matplotlib.pyplot as plt

domain = gym.make('MountainCar-v0')


# Q-Learning settings
LEARNING_RATE = 0.1    # between 0 and 1, 0 means Q values are never updated, therefore nothing is learned, higher value  = quick learning
DISCOUNT = 0.95     # measure of how much to care about future reward (between 0 and 1, generally high)
EPISODES = 4000   # number of iterations of sim would run
SHOW_EVERY = 3000   # since we cannot render the domain everytime, so render it every SHOW_EVERY episodes

DISCRETE_OS_SIZE = [40] * len(domain.observation_space.high)
discrete_os_win_size = (domain.observation_space.high - domain.observation_space.low) / DISCRETE_OS_SIZE

# for tracking various parameters and graph them
eps_rewards = []
aggr_eps_rewards = {'eps':[], 'avg': [], 'max': [], 'min': []}


# Exploration settings
epsilon = 1     # not a constant, qoing to be decayed , we cannot move initially based on maximum q value because these moves based on max q values are worthless
START_EPSILON_DECAYING = 1   # so we use epsilon to move randomly at initial
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low = -2, high=0, size=(DISCRETE_OS_SIZE + [domain.action_space.n]))   # q-table initialized randomly 


def get_discrete_state(state):   # the function to convert the domain state from continuous to discrete state, for eg, convert position value from 2.10035647 to 2.1
    discrete_state = (state - domain.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))     # we use this tuple to look up the 3 Q values for the available actions in the q-table



for episode in range(EPISODES):  # loop for number of iterations
    episode_reward = 0   # to store the reward info
    discrete_state = get_discrete_state(domain.reset())   # get the initial state values from reset
    done = False    # Flag to indicate if simulation end 

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:   # if simulation did not end after last step
        if np.random.random() > epsilon:   # randomly pick a number between 0 and 1, if greater than epsilon, take action based on max q value else move randomly
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])   # we take the action based on maximum q value from table for the discrete state value
        else:
            action = np.random.randint(0, domain.action_space.n)


        new_state, reward, done, _ = domain.step(action)
        episode_reward += reward    #  store the reward info

        new_discrete_state = get_discrete_state(new_state)  # after taking the action,  get the new discrete state from new state from that action

        if episode % SHOW_EVERY == 0:
            domain.render()
        
        

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])   # best q value for the action that have already been taken, the reward gets back propagated and update the previous q value.

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            #  equation for new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value for current state and action
            q_table[discrete_state + (action,)] = new_q


        # if Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif  new_state[0] >= domain.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state   # reset the discrete state variable
    

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value   # decay epsilon value every episode

    eps_rewards.append(episode_reward)   # for tracking parameters
    if not episode % SHOW_EVERY:
        average_reward = sum(eps_rewards[-SHOW_EVERY:]) / SHOW_EVERY
        aggr_eps_rewards['ep'].append(episode)
        aggr_eps_rewards['avg'].append(average_reward)
        aggr_eps_rewards['max'].append(max(eps_rewards[-SHOW_EVERY:]))
        aggr_eps_rewards['min'].append(min(eps_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    if episode % 100 == 0:
        np.save(f'E:\edx\harvard cs\Cs 50 AI\Deep Learning Specialization\Reinforcement Learning Sentdex\q_tables/ {episode}-qtable.npy', q_table)  # save each 100th episode's q-table

domain.close()   # close the domain

# Plot the graph of parameters
plt.plot(aggr_eps_rewards['ep'], aggr_eps_rewards['avg'], label = 'average rewards')
plt.plot(aggr_eps_rewards['ep'], aggr_eps_rewards['max'], label = 'max rewards')
plt.plot(aggr_eps_rewards['ep'], aggr_eps_rewards['min'], label = 'min rewards')
plt.legend(loc = 4)
plt.show()

