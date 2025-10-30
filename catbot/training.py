import random
from typing import Dict, List, Tuple
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

# TODO: Remove Tuple and List return from final submission
def train_bot(cat_name, render: int = -1) -> Tuple[Dict[int, np.ndarray], List[int]]:
    # TODO: Override instructions and add global declaration
    global current_epsilon, current_alpha

    env = make_env(cat_type=cat_name)

    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000  # Training is capped at 5000 episodes for this project

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
        #############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                      #
        #############################################################################
        # Hint: These are the general steps you must implement for each episode.    #
        # 1. Reset the environment to start a new episode.                          #
        # 2. Decide whether to explore or exploit.                                  #
        # 3. Take the action and observe the next state.                            #
        # 4. Since this environment doesn't give rewards, compute reward manually   #
        # 5. Update the Q-table accordingly based on agent's rewards.               #
        #############################################################################

        # 1. Reset environment
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            # 2. Decide whether to explore or exploit
            if random.random() < current_epsilon:
                action = env.action_space.sample()  # Explore
            else:
                # Exploit: Randomly break ties
                max_q = np.max(q_table[obs])
                best_actions = np.where(q_table[obs] == max_q)[0]
                action = random.choice(best_actions)

            # 3. Take action and observe
            new_obs, _, terminated, truncated, info = env.step(action)
            steps += 1

            # 4. Compute reward manually
            reward = 0.0
            if terminated:
                reward = 100.0  # Cat caught
            elif env.cat.current_distance < env.cat.prev_distance:
                reward = 1.0  # Got closer
            elif env.cat.current_distance > env.cat.prev_distance:
                reward = -5.0  # Moved farther
            else:
                reward = -0.5  # Stalled

            # 5. Update Q-table (using the dictionary)
            old_value = q_table[obs][action]
            next_max = np.max(q_table[new_obs])

            # Q-Learning Formula
            new_value = (1 - current_alpha) * old_value + current_alpha * (reward + GAMMA * next_max)
            q_table[obs][action] = new_value

            obs = new_obs
            done = terminated or truncated

        # --- End of while loop ---

        steps_per_episode.append(steps)

        # Decay hyperparameters
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
        current_alpha = max(MIN_ALPHA, current_alpha * ALPHA_DECAY)

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02,
                         window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    # TODO: Remove steps_per_episode from final submission
    return q_table, steps_per_episode