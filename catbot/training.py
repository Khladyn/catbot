import random
from typing import Tuple, List
import numpy as np
from utility import play_q_table
from cat_env import make_env

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Hyperparameters for Q-Learning
ALPHA = 0.1  # Learning Rate (Will be decayed)
GAMMA = 0.99  # Discount Factor
EPSILON = 1.0  # Initial Exploration Rate
EPSILON_DECAY = 0.9995  # Decay rate per episode
MIN_EPSILON = 0.01  # Minimum Epsilon value
ALPHA_DECAY = 0.9999  # Decay rate for Learning Rate (Optimization)
MIN_ALPHA = 0.001  # Minimum Alpha value

# Global/Tracking Variables
current_epsilon = EPSILON
current_alpha = ALPHA


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

# Updated return type hint: Q-table (np.ndarray) and steps_per_episode (List[int])
def train_bot(cat_name, render: int = -1) -> Tuple[np.ndarray, List[int]]:
    global current_epsilon, current_alpha
    env = make_env(cat_type=cat_name)

    # 2. OPTIMIZATION: Initialize Q-table as a single NumPy array (225 states x 4 actions)
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    q_table = np.zeros((state_space_size, action_space_size))

    # Training hyperparameters
    episodes = 5000

    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################

    # Reset tracking variables for this run
    current_epsilon = EPSILON
    current_alpha = ALPHA
    steps_per_episode = []

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            # 2. Decide whether to explore or exploit (Epsilon-Greedy)
            if random.random() < current_epsilon:
                action = env.action_space.sample()  # Explore (Random action)
            else:
                # 1. REDUCE OSCILLATION: Implement Random Tie-Breaking
                max_q = np.max(q_table[obs])

                # Get the indices (actions) that equal the max Q-value.
                best_actions = np.where(q_table[obs] == max_q)[0]

                # Randomly choose one of the best actions.
                action = random.choice(best_actions)
                # --------------------------------------------------

            # 3. Take the action and observe the next state.
            new_obs, _, terminated, truncated, info = env.step(action)
            steps += 1

            # 4. SIMPLIFICATION/REWARDS: Manual reward calculation
            reward = 0.0

            if terminated:
                # Cat caught! High positive reward
                reward = 100.0
            elif env.cat.current_distance < env.cat.prev_distance:
                # Got closer to the cat
                reward = 1.0
            elif env.cat.current_distance > env.cat.prev_distance:
                # Moved farther away (Severe Penalty)
                reward = -5.0
            else:
                # Distance is the same (Penalty for stalling/shaking)
                reward = -0.5

            # 5. Update the Q-table (Q-Learning Formula)
            old_value = q_table[obs, action]  # 2. OPTIMIZATION: NumPy array indexing
            next_max = np.max(q_table[new_obs])  # 2. OPTIMIZATION: NumPy array indexing

            # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * [reward + gamma * max(Q(s', a'))]
            new_value = (1 - current_alpha) * old_value + current_alpha * (reward + GAMMA * next_max)
            q_table[obs, action] = new_value  # 2. OPTIMIZATION: NumPy array indexing

            # Set current observation to new observation
            obs = new_obs
            done = terminated or truncated

        # Log the steps taken for this episode
        steps_per_episode.append(steps)

        # Decay Hyperparameters after the episode ends
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        # 4. Q-Learning Core: Decay Alpha
        current_alpha = max(MIN_ALPHA, current_alpha * ALPHA_DECAY)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # 🛑 Print status to show core metrics 🛑
        print(
            f'Episode {ep}/{episodes}: Epsilon {current_epsilon:.3f}, Alpha {current_alpha:.4f}, Steps {steps}')

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)

            steps_taken = play_q_table(
                viz_env, q_table, max_steps=100, move_delay=0.02,
                window_title=f"{cat_name}: Training Episode {ep}/{episodes} (Epsilon: {current_epsilon:.3f})"
            )

            print(f'[RENDERED] episode {ep} finished. Steps: {steps_taken}')

    # Return the Q-table (NumPy array) and the steps data
    return q_table, steps_per_episode