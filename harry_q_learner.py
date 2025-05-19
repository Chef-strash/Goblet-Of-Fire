import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque

# Maze Layout (1 = wall, 0 = path) (V1 layout)
MAZE=[
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

TILE_SIZE= 50
GRID_WIDTH= 15
GRID_HEIGHT= 10

actions= [(-1, 0), (1, 0), (0, -1), (0, 1)]       # Left, Right, Up, Down

class QlearningAgent:
    def __init__(self,learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        # Q-table initialization, all values are set to 0
        self.q_table= np.zeros(( GRID_HEIGHT, GRID_WIDTH, len(actions)))

        # Hyperparameters
        self.learning_rate= learning_rate
        self.discount_factor= discount_factor              # Discount factor for future rewards
        self.exploration_start= exploration_start          # Initial exploration rate
        self.exploration_end= exploration_end              # Final exploration rate
        self.num_episodes= num_episodes
    
    def get_exploration_rate(self,current_episode):
        # Exploration rate for "ε-greedy strategy"
        exploration_rate= self.exploration_end + (self.exploration_start - self.exploration_end) * np.exp(- current_episode / self.num_episodes)
        return exploration_rate

    def get_valid_actions(self, state, current_episode):
        exploration_rate= self.get_exploration_rate(current_episode)

        # Filter valid actions
        x, y = state
        valid_actions = []
        for idx, (dx, dy) in enumerate(actions):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and MAZE[new_y][new_x] == 0:
                valid_actions.append(idx)

        # This is the "ε-greedy strategy", pivital role in early training
        if np.random.rand() < exploration_rate:
            return np.random.choice(valid_actions)
        
        else:
            # Among valid actions, pick the one with highest Q-value
            q_values = self.q_table[y, x, valid_actions]
            best_idx = np.argmax(q_values)
            return valid_actions[best_idx]

    def update_q_value(self, state, action, next_state, reward):

        current_q_value = self.q_table[state[1], state[0], action]
        best_future_q_value = np.max(self.q_table[next_state[1], next_state[0]])

        # Updation through Bellman equation
        new_q_value=(1-self.learning_rate)*current_q_value + self.learning_rate*(reward + self.discount_factor*best_future_q_value)
        self.q_table[state[1], state[0], action] = new_q_value

# Function for the Reward System
def get_reward(current_state, next_state, death_eater_pos, cup_pos):
    goal_reward= 100
    death_eater_penalty= -40
    move_reward= 2
    
    is_done= False
    reward= 0

    # Reward logic
    if next_state==cup_pos:
        reward= goal_reward                            # Reward for motivation the Agent to reach the cup
        is_done= True 
    elif next_state==death_eater_pos:
        reward= death_eater_penalty                    # Penalty for being caught by the Death Eater
        is_done= True
    elif next_state == current_state:
        reward= -5                                     # Penalty for not moving
    elif(MAZE[next_state[1]][next_state[0]]==1):
        reward= -5                                     # Penalty for hitting the wall 
    else:
        reward+= move_reward                           # Reward for taking a step 
    
    # Penalty to walk towards the Death Eater
    distance= (next_state[0]-death_eater_pos[0])**2 + (next_state[1]-death_eater_pos[1])**2
    if distance<= 9:
        reward+= -20+(distance+1)
    elif distance > 9 and distance < 25:
        reward+= 10
    elif distance >= 25:
        reward= 0
    
    return reward, is_done
    
# Generating random cell
def gen_random_cell(exclude=[]):  # Exclude is a list of cells to be avoided apart from the wall cell
    while True:
        x=random.randint(0, GRID_WIDTH - 1)
        y=random.randint(0, GRID_HEIGHT - 1)
        if MAZE[y][x]==0 and (x,y) not in exclude:  # (x,y) are co-ordinates of the patches but not the cellss to be excluded 
            return (x, y)

# Traversal Mechanism of Death Eater (BFS)
def bfs(start, goal):
    queue= deque()
    queue.append((start, []))
    visited= set()
    visited.add(start)

    while queue:
        (x, y), path= queue.popleft()
        if (x, y) == goal:
            return path + [(x, y)]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and MAZE[ny][nx] == 0:
                if (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(x, y)]))
                    visited.add((nx, ny))
    return []

# Function to simulate an episode that one generation of the Agent 
def finish_episode(Harry, harry_pos, death_eater_pos, cup_pos, current_episode, train=True):

    current_state= harry_pos
    is_done= False
    episode_reward= 0
    episode_step= 0
    path=[current_state]
  
    # Initiating the episode
    while not is_done:
        # If the agents takes more than these no. of steps, it is considered a failure as their are only 78 free cells,
        # so if it take more than 100 steps, it means the agent is stuck somewhere
        if episode_step >= 100:  
            break

        action= Harry.get_valid_actions(current_state, current_episode)

        # Harry moves
        nx = current_state[0] + actions[action][0]
        ny = current_state[1] + actions[action][1]
 
        # Check within grid and not hitting wall
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and MAZE[ny][nx] == 0:
            next_state = (nx, ny)
        else:                                                                           
            next_state = current_state  # Harry stays in place if move is invalid

        # Death eater moves
        DE_path=bfs(death_eater_pos, harry_pos)
        if len(DE_path) > 1:
            death_eater_pos= DE_path[1]

        # Reward for the action taken
        reward, is_done= get_reward(current_state, next_state, death_eater_pos, cup_pos)
        
        episode_reward+= reward
        episode_step+= 1

        if train == True:
            Harry.update_q_value(current_state, action, next_state, reward)
        path.append(next_state)
        current_state= next_state
    
    return episode_reward, episode_step, path

# Function to train the agent
def train_Harry(Harry, harry_pos, death_eater_pos, cup_pos, num_episodes):
    episode_rewards= []
    episode_steps= []

    win= 0
    current_streak= 0
    max_streak= 0
    generations_to_10_streak= None
    target_streak= 10

    for episode in range(num_episodes):
        harry_pos= gen_random_cell([cup_pos])
        death_eater_pos= gen_random_cell([harry_pos, cup_pos])
        episode_reward, episode_step, path= finish_episode(Harry, harry_pos, death_eater_pos, cup_pos, episode)

        # Update reward and steps
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        # Track success
        if cup_pos in path:
            win  += 1
            current_streak+= 1
            if current_streak == target_streak and generations_to_10_streak is None:
                generations_to_10_streak= episode + 1  # +1 because the episode number starts from 0
        else:
            current_streak= 0

        max_streak= max(max_streak, current_streak)

    # Training Summary
    avg_reward = sum(episode_rewards) / num_episodes
    avg_steps = sum(episode_steps) / num_episodes

    print("\n📊 TRAINING SUMMARY 📊")
    print(f"🏆 Total wins: {win} out of {num_episodes}")
    print(f"🎯 Number of generations it took to win 10 consequitve times: {generations_to_10_streak if generations_to_10_streak else 'Not achieved'}")
    print(f"📈 Maximum win streak during training: {max_streak}")
    print(f"💰 Average reward: {avg_reward:.2f}")
    print(f"🚶 Average steps: {avg_steps:.2f}")

    # Plotting the Graphs
    plt.figure(figsize=(10, 5))

    # Reward per Episode (Line Plot)
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward', color='blue', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.legend()

    # Steps per Episode (Line Plot)
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps, label='Steps', color='orange', linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.title('Steps per Episode')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_summary_plot.png")

# Initiliing the positions of the Agent and the Death Eater
cup_pos=(7,5)                                                 #position of the cup
harry_pos=gen_random_cell([cup_pos])                          #position of harry
death_eater_pos=gen_random_cell([harry_pos,cup_pos])          #position of death eater


# Initilizing the Q-learning agent as Harry and training it 
Harry=QlearningAgent()
train_Harry(Harry, harry_pos, death_eater_pos, cup_pos, num_episodes=20000)

with open("Qcfinal_table.pkl", "wb") as f:
        pickle.dump(Harry.q_table, f)

print("Qcfinal-table saved!")
