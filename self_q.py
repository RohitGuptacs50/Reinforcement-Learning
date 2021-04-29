import numpy as np

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle   # to save and load q_tables
from matplotlib import style
import time

from numpy.__config__ import show

style.use('ggplot')

SIZE = 10   # size of the domain grid 10x10 square, the larger it is, larger is the q table, will take more memory and more time for the model to learn
HM_EPISODES = 25000  # number of episodes
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.5
EPS_DECAY = 0.9999    # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through domain visually.

start_q_table = None    # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95


PLAYER_N = 1  # Player key in dict
FOOD_N = 2  # FOod key in dict
ENEMY_N = 3   # enemy key in dict


# dictionary for colors
d = {
    1: (255, 175, 0),      # blueish color
    2: (0, 255, 0),      # Green
    3: (0, 0, 255)       # Red
}


class Blob:   # domain consist of squares (or) blobs
    def __init__(self): # initialize blobs randomly
        self.x = np.random.randint(0, SIZE)   # give random integer from 0(inclusive) to SIZE(exclusive)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):   # for debugging
        return f'{self.x}, {self.y}'

    def __sub__(self, other):    # 'other' is any other blob with x and y attributes
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):      ## All are diagonal movements
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        if choice == 2:
            self.move(x = -1, y = 1)
        if choice == 3:
            self.move(x = 1, y = -1)

    
    def move(self, x = False, y = False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        
        ## If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        
        # if we are out of bounds(or grid), fix(control for attempts for the blob to go out of bounds.)
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


player = Blob()
food = Blob()
enemy = Blob()


print(player)
print(food)
print(player - food)
player.move()
print(player - food)
player.action(2)
print(player - food)


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):   # -9 to 9 inclusive
        for y1 in range(-SIZE + 1, SIZE):   # # -9 to 9 inclusive
            for x2 in range(-SIZE + 1, SIZE):   # -9 to 9 inclusive
                for y2 in range(-SIZE + 1, SIZE):# -9 to 9 inclusive
                    q_table[(x1, y1), (x2, y2)] = [np.random.uniform(-5, 0) for i in range(4)] 

else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)

print(q_table[(-9, 2), (3, 9)])


episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()  # for each new episode. reinitialize the blobs
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f'on #{episode}, epsilon is {epsilon}')
        print(f'{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        # print(obs)
        if np.random.random() > epsilon:
            # Get the action
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action
        player.action(action)

        #### MAYBE ###   to move other objects
        #enemy.move()
        #food.move()
        ##############
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY


        new_obs = (player - food, player - enemy)    # new observation
        max_future_q = np.max(q_table[new_obs])     # # max Q value for this new obs
        current_q = q_table[obs][action]     # current Q for our chosen action

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)   # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]    # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]      # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]      # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')    # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))    # resizing so we can see our agent in all its glory.
            cv2.imshow('image', np.array(img))    # show it
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:   # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break
    

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY



moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = 'valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'Reward {SHOW_EVERY}ma')
plt.xlabel('episode #')
plt.show()


with open(f'qtable-{int(time.time())}.pickle', 'wb') as f:
    pickle.dump(q_table, f)

