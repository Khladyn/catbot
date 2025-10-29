## 🐾 Q-Learning Agent: Cat Bot

This repository contains the implementation of a reinforcement learning (RL) agent trained via **Q-Learning** to efficiently catch various types of cats on a discrete grid environment. The project focuses on creating a robust learning pipeline, leveraging a generalized state space and manual reward shaping for complex pathfinding challenges.

-----

## 💡 Overview

The objective is to train a bot to find the optimal policy (a sequence of actions) to minimize the steps required to reach and "capture" the target cat. The agent learns from discrete rewards and penalties assigned for actions like closing distance, retreating, and stalling.

The training pipeline implements a flexible environment where the bot's policy must generalize to different cat behaviors (e.g., passive, random, evasive) defined within the environment module (`cat_env.py`).

-----

## ✨ Features and Technical Implementation

### 1\. Core Q-Learning Architecture

  * **Algorithm:** **Off-Policy Temporal Difference (TD)** learning using the **Q-Learning** algorithm.
  * **Policy:** **$\epsilon$-Greedy Exploration** is used to balance exploring new actions with exploiting the best known Q-values.
  * **Generalized State Space:** The agent uses a **relative observation space** (a $\mathbf{15 \times 15}$ grid of $\mathbf{225}$ states). Instead of tracking absolute coordinates, the state represents the cat's position **relative** to the agent. This allows the learned policy to be generalized across the entire map.

### 2\. Foundational Reward Shaping

A heuristic reward function is implemented to guide the agent's behavior:

| Condition | Reward ($\mathbf{R}$) | Technical Term | Purpose |
| :--- | :--- | :--- | :--- |
| **Capture** | $+100.0$ | Final Positive Reinforcement | Primary Goal Reward. |
| **Move Closer** | $+1.0$ | Immediate Positive Reinforcement | Encourages closing the **Manhattan Distance**. |
| **Stall** | $-0.5$ | Stalling Penalty | Discourages wasting time by taking an action that doesn't change distance. |
| **Retreat** | $-5.0$ | Severe Penalty | Strongly discourages moving farther away from the cat. |

### 3\. Hyperparameters and Training Loop

The training is controlled by standard RL parameters defined in `training.py`:

  * **Learning Rate ($\alpha$):** $\mathbf{0.1}$ (Determines how quickly new information overwrites old Q-values).
  * **Discount Factor ($\gamma$):** $\mathbf{0.99}$ (Agent is mostly long-sighted, prioritizing the final capture reward).
  * **Exploration Decay:** $\epsilon$ gradually decays from $\mathbf{1.0}$ to $\mathbf{0.01}$ over the training episodes.

### 4\. Telemetry and Logging (For Debugging)

To facilitate visualization and analysis, crucial tracking capabilities are included:

  * **Episode Steps Tracking:** The final training loop returns the **steps per episode**, a direct measure of **policy efficiency** and convergence speed.
  * **Console Enhancement:** Console logs now display the current **Epsilon** ($\epsilon$) value and the **Steps Taken** for visualized episodes, providing clear feedback on the agent's learning progression.

-----

## 📂 Repository Structure (Implied)

| File | Function |
| :--- | :--- |
| **`training.py`** | Contains the $\mathbf{Q-Learning}$ algorithm, training loop, all $\alpha / \gamma$ hyperparameters, and reward shaping logic. |
| **`cat\_env.py`** | Defines the environment, grid dimensions, state observation logic, and the behavior of the different cat types (e.g., `PaotsinCat`). |
| **`utility.py`** | Includes helper functions for visualizing the trained policy (e.g., `play_q_table`). |
| **`bot.py`** | Main entry point for running the training and visualization process. |

-----

## ▶️ Setup and Running

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Khladyn/mco2_catbot.git
    cd mco2_catbot
    ```

2.  **Run Training:** Execute the main bot script, specifying the cat type.

      * **To run without visualization (faster training):**
        ```bash
        python bot.py --cat [cat-type]
        ```
      * **To run *with* visualization (rendering):** Use the `--render` flag followed by the episode interval. This plays back an episode every 'N' episodes.
        ```bash
        python bot.py --cat [cat-type] --render [N] 
        # Example: Render every 50 episodes
        # python bot.py --cat paotsin --render 50
        ```

3.  **Analyze Results:** After training is complete, the logged data (including `steps_per_episode`) can be used to plot learning curves and assess the agent's performance.
