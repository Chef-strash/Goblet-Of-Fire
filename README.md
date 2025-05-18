# 🧠 Goblet of Fire: Q-Learning Based Maze Escape 🧙‍♂️🔥

This project uses **Q-learning** to train an agent (Harry Potter) to escape a maze, avoid a pursuing **Death Eater**, and reach the **Triwizard Cup**. The environment is visualized with **Pygame**, and the agent learns through trial and error.

---

## 📁 About the Repo

This repository is a submission for the **Secretory Recruitment Task'25** of **BSC, SnT**.  
The assigned problem is titled **"Goblet of Fire"**, inspired by the magical world of Harry Potter.

---

## 📌 Assumptions

- 🟩 Maze walls are shown in **green**, traversable paths in **white**.
- 🔵 **Harry (the agent)** is shown in **blue**; 🔴 the **Death Eater** is shown in **red**.
- The **map layout** is static and referred to as **Map V1**.
- 🏆 The **Triwizard Cup** is placed at a fixed location: **(7, 5)**.
- Harry and the Death Eater move on a **grid**, one cell per step.
- The Death Eater **knows Harry's location** and moves intelligently using **Breadth-First Search (BFS)**.
- Spawn positions for Harry and the Death Eater are **randomized** at the start of each episode.
- Harry can **sense** the Death Eater if it comes within a **3-cell radius**, even through walls, influencing his decision-making.

---

## 🔍 Approach

The project is structured into **three major components**:

### 1. 🧩 Maze Generation with Pygame

- The maze environment is created using **Pygame**, enabling visual feedback.
- A **15x10 grid** defines the maze, where each tile is **50x50 pixels**.
- Color coding helps distinguish walls, paths, Harry, the Death Eater, and the cup.
- The cup remains at a fixed location, while Harry and the Death Eater spawn randomly on open tiles.

### 2. 🤖 Q-Learning Algorithm (Model-Free RL)

Q-learning is used to teach Harry the optimal path to reach the cup while avoiding the Death Eater. Key features:

- The **Q-table** maps each (state, action) pair to a Q-value.
- **States** are defined by the agent’s (x, y) grid location.
- **Actions** correspond to the four possible moves: up, down, left, right.
- An **ε-greedy strategy** balances exploration and exploitation. The exploration rate decreases over time, encouraging learning and then optimization.
- **Temporal Difference (TD) Learning** is used to update Q-values based on the Bellman Equation.
- Learning continues for a large number of episodes, allowing the agent to converge on a near-optimal policy.

#### Highlights:
- Modular class-based design for the Q-learning agent.
- Dynamic ε-decay for better exploration/exploitation balance.
- Trained over **100,000 episodes** for robustness.

### 3. 🎯 Reward System Design

The reward system is carefully structured to guide Harry’s learning toward desirable behavior:

- **Positive Reward**: For reaching the cup, proportional to proximity.
- **Negative Reward**: 
  - Heavy penalty for being caught by the Death Eater.
  - Small penalty for staying in the same position or hitting a wall.
- **Movement Incentive**: Slight reward for valid moves to encourage exploration.
- **Proximity Awareness**: If Harry moves closer to the Death Eater (within 3-cell radius), additional penalties are applied—even if the Death Eater is behind a wall. This emulates a "fear response" to unseen danger.

This reward function creates a **balance between exploration, goal-seeking, and danger avoidance**. It is tuned to teach Harry to reach the goal while prioritizing safety.

---

## 📊 Evaluation Metrics

To measure performance, we track:

- 🔁 **Generations to reach 10 consecutive successful escapes** (i.e., reaching the cup).
- 🏆 **Maximum escape streak** achieved during training.
- 📉 **Cumulative reward and steps per episode** plotted and saved to file.

These metrics provide insight into both **efficiency** and **stability** of the learned behavior.

---

## 🚀 How to Run

Install dependencies and run:

```bash
pip install pygame numpy matplotlib
python main.py
