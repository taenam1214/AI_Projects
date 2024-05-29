import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while not done:
            # Implementing ε-greedy policy
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: select a random action
            else:
                # Exploit: select the action with max Q-value for the current state
                action = max(list(range(env.action_space.n)), key=lambda x: Q_table[(obs, x)])

            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get max Q-value for the new state
            max_future_q = max([Q_table[(new_obs, a)] for a in range(env.action_space.n)])

            # Compute update rule
            if not done:
                # Normal update rule
                current_q = Q_table[(obs, action)]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
                Q_table[(obs, action)] = new_q
            else:
                # Update for terminal state
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * reward

            obs = new_obs  # move to the next state
            episode_reward += reward  # update episode reward

        # Decay epsilon
        EPSILON *= EPSILON_DECAY
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################