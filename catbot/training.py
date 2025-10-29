import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Hyperparameters for Q-Learning
ALPHA = 0.1  # Learning Rate
GAMMA = 0.99  # Discount Factor
EPSILON = 1.0  # Initial Exploration Rate
EPSILON_DECAY = 0.9995  # Decay rate per episode
MIN_EPSILON = 0.01  # Minimum Epsilon value

# You can use a global or dictionary to store the current epsilon,
# but for simplicity within the scope of this function, we'll
# use a single variable for decay across episodes.
current_epsilon = EPSILON


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    global current_epsilon
    env = make_env(cat_type=cat_name)

    # 🛑 CORRECTED: Initialize Q-table with the correct, dynamic size (225 states)
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(env.observation_space.n)
    }

    # Training hyperparameters
    episodes = 5000  # Training is capped at 5000 episodes for this project

    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################

    # Initialize epsilon for this training run
    current_epsilon = EPSILON

    # 🛑 ADDED: Initialize the list to store steps per episode
    steps_per_episode = []

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################

    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################

        obs, info = env.reset()
        done = False

        # 🛑 ADDED: Step counter for the current episode
        steps = 0

        while not done:
            # 2. Decide whether to explore or exploit (Epsilon-Greedy)
            if random.random() < current_epsilon:
                action = env.action_space.sample()  # Explore (Random action)
            else:
                action = np.argmax(q_table[obs])  # Exploit (Best known action)

            # 3. Take the action and observe the next state.
            new_obs, _, terminated, truncated, info = env.step(action)

            # 🛑 ADDED: Increment the step counter
            steps += 1

            # 4. Since this environment doesn't give rewards, compute reward manually
            reward = 0.0

            if terminated:
                # Cat caught! High positive reward
                reward = 100.0
            elif env.cat.current_distance < env.cat.prev_distance:
                # Got closer to the cat (Manhattan distance decreased)
                reward = 1.0
            elif env.cat.current_distance > env.cat.prev_distance:
                # Moved farther away from the cat (Severe Penalty)
                reward = -5.0  # Increased penalty to strongly discourage retreat
            else:
                # Distance is the same (Penalty for stalling/shaking)
                reward = -0.5

            # 5. Update the Q-table (Q-Learning Formula)
            old_value = q_table[obs][action]
            next_max = np.max(q_table[new_obs])

            # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * [reward + gamma * max(Q(s', a'))]
            new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            q_table[obs][action] = new_value

            # Set current observation to new observation
            obs = new_obs
            done = terminated or truncated

        # 🛑 ADDED: Log the steps taken for this episode
        steps_per_episode.append(steps)

        # Decay Epsilon after the episode ends
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)

            # 🛑 CAPTURE THE RETURNED STEP COUNT HERE 🛑
            steps_taken = play_q_table(
                viz_env, q_table, max_steps=100, move_delay=0.02,
                window_title=f"{cat_name}: Training Episode {ep}/{episodes} (Epsilon: {current_epsilon:.3f})"
            )

            # 🛑 UPDATE THE PRINT STATEMENT 🛑
            print(f'episode {ep} finished. Epsilon: {current_epsilon:.3f}. Steps: {steps_taken}')

    # 🛑 Must return the steps data along with the Q-table
    return q_table, steps_per_episode