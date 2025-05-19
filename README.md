# 🧙‍♂️ Goblet of Fire — Reinforcement Learning Agent (Q-Learning Based)

## 📌 About the Repo

This repository is a submission for the **Secretary Recruitment Task'25** of the **Brain and Cognetive Science Club, SnT**.

This particular challenge is titled **"Goblet of Fire"** — where the goal is to train an agent (Harry Potter) to find the **Triwizard Cup** in a maze-like environment while avoiding being capture by a pursuing **Death Eater**.

---

### 📁 Project Structure

| **File**                   | **Purpose**                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| `GOF_auto_play.py`         | Rendered game environment played with the help of the trained Q-table       |
| `GOF_manuel_play.py`       | Game environment for manual play (user-controlled Harry)                    |
| `harry_q_learner.py`       | Implementation of the Q-Learning algorithm                                  |
| `Qc3_table.pkl`            | Saved Q-table after training (used for evaluation and auto-play)            |
| `training_plot3c.png`      | Plot of rewards and steps taken per episode during training                 |
| `README.md`                | This documentation file                                                     |


---

## 📜 Assumptions and Rendering Deatils

- 🟩 Walls are **green** in color.
- ⚪ Traversable paths are **white**.
- 🔵 Harry Potter (agent) is represented in **blue**.
- 🔴 The Death Eater is shown in **red**.
- ⭐ The Triwizard Cup has a fixed position at **(7, 5)** (X, Y).
- The environment is based on a fixed **15x10 grid-based map (Map Version V1)**.
- Both Harry and the Death Eater move one cell at a time in a **grid-block system**.
- The **Death Eater always knows Harry's position** and moves using **Breadth-First Search (BFS)**, simulating a magical sensing ability.
- Both characters spawn randomly on valid positions at the start of each episode.
- When the Death Eater is within **3 blocks**, Harry can "sense" its presence, even if obstructed by walls.
- How to see Harry in Action
  - Harry can be see in action inthe rendered environment with the Help of **GOF_auto_play.py**, quided by the Learned Policy from the Q Table.
---

## 🔍 Approach

The problem is broken down into three major components:

---

### 1. 🧩 Maze Generation with Pygame

- The maze is a **hardcoded 2D matrix** where `1` denotes walls and `0` denotes free space.
- Visualization is done using **Pygame**, where:
  - Walls are rendered as green blocks.
  - Open paths are white.
  - The agent and enemies are colored according to the legend above.
- The Cup's position is fixed and static; however, **Harry and the Death Eater are placed randomly** in traversable cells at the start of each episode.
- The Maze was rendered with the help of "pygame" library, the File **GOF_manuel_play.py** contains the rendered environment aided with user-playthrough using the Navigation.
- The file **GOF_auto_play.py** is integrated with the Trained Q_table, the script fetches the action with highest q value for a given state and navigates the maze.
---

### 2. 🤖 Q-Learning Algorithm 

The heart of this project lies in its use of the **Q-Learning algorithm**, implemented in a class-based, modular way that facilitates clarity and control. This Q-learning algorithm does not make use of a Neural Network.
This algorithm is shaped by it's **Reward System** for interacting with the environment and **The Bellman Equation** for updating the weights.
- The Q-learning Algorithm is present in **harry_q_learner.py**.

#### 🧠 Learning Setup

A class called `QlearningAgent` is created to encapsulate all learning behavior. Upon initialization, it sets up a 3D Q-table with dimensions `[grid_height][grid_width][4]`, one for each action (up, down, left, right). Hyperparameters such as **learning rate**, **discount factor**, and **exploration parameters** are also initialized here.
- Learning rate = 0.1
- discount factor = 0.9
- exploration_start=1.0, exploration_end=0.01 (These two help define the exploration rate)

#### 🎲 Exploration Strategy

Each time the agent must act, it decides between **exploration and exploitation** using an ε-greedy strategy:
- With a decreasing ε over time, Harry starts out exploring more and gradually begins to **rely on what he's learned**.
- exploration_rate = exploration_start - (exploration_start - exploration_end) * (current_episode / num_episodes)
- `get_exploration_rate()` returns a linearly decaying exploration rate.
- `get_valid_actions()` either selects a random move (exploration) or chooses the highest Q-value move from the Q-table (exploitation).

#### 📈 Learning Mechanism

As Harry moves through the maze:
- After each move, `update_q_value()` updates the Q-table using the **Bellman Equation**:
  
  Q(s, a) (1 - learning rate)Q(s, a) + alpha( r + (discount_factor) * max[Q(s', a')] )
  

- `finish_episode()` simulates one full episode, where:
  - Harry chooses moves using Q-values.
  - The Death Eater chases using **BFS**, always moving toward Harry.
  - The custom reward function (`get_reward()`) evaluates each step.
  - If Harry reaches the cup or is caught, the episode ends.

#### 🧪 Training Loop

Training is orchestrated by `train_Harry()`, which:
- Runs thousands of episodes.
- Tracks total **reward** and **steps** taken per episode.
- **Evaluates when Harry first survives 10 consecutive successful escapes**, to measure learning convergence.
- Automatically plots performance using Matplotlib.

---

### 3. 🏆 Reward System

A fine-tuned **reward function** was key to teaching Harry effective strategies for survival and success.

#### 📜 `get_reward()` Logic:

| Scenario                            | Reward     | Purpose                                              |
|-------------------------------------|------------|------------------------------------------------------|
| Reaches the Cup                     | `+100/(dist+1)` | Encourages getting closer to goal; scales with distance |
| Gets caught by Death Eater          | `-40`      | Strong penalty for failure                          |
| Tries to walk into wall or stays    | `-5`       | Discourages invalid moves                           |
| Valid step in any direction         | `+2`       | Encourages movement and exploration                 |
| Near Death Eater (≤ 3 blocks)       | `-20 + (distance + 1)` | Simulates danger detection and fear                |
| Far from Death Eater (> 9 blocks)   | `+10`      | Rewards maintaining a safe distance                 |

This blend of rewards and penalties helped the agent **balance exploration**, **goal-seeking**, and **threat avoidance**, creating a more natural and responsive behavior during training.

---

## 📊 Performance Monitoring

- During training, reward and step plots are generated and saved (`training_plot6c.png`).
- Evaluation is performed using `evaluate_Harry()` to check when Harry achieves a **10-win streak** and to measure his **longest survival streak**.

