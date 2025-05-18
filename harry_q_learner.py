import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import random

#Maze Layout (1 = wall, 0 = path) (V1 layout)
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

TILE_SIZE = 50
GRID_WIDTH = 15
GRID_HEIGHT = 10

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

class QlearningAgent:
    def __init__(self,learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.q_table=np.zeros(( GRID_HEIGHT, GRID_WIDTH, len(actions)))
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.exploration_start=exploration_start
        self.exploration_end=exploration_end
        self.num_episodes=num_episodes
    
    def get_exploration_rate(self,current_episode):
        #exploration_rate= self.exploration_start*(self.exploration_end/self.exploration_start)**(current_episode/self.num_episodes)
        exploration_rate=self.exploration_start - (self.exploration_start - self.exploration_end) * (current_episode / self.num_episodes)
        return exploration_rate
    
    def get_valid_actions(self, state, current_episode):
        exploration_rate=self.get_exploration_rate(current_episode)
        if np.random.rand()<exploration_rate:     #see this
            return np.random.choice(len(actions)) 
        else:
            return np.argmax(self.q_table[state[1], state[0]])

    
    def update_q_value(self, state, action, next_state, reward):

        current_q_value = self.q_table[state[1], state[0], action]

        best_future_q_value = np.max(self.q_table[next_state[1], next_state[0]])

        new_q_value=(1-self.learning_rate)*current_q_value + self.learning_rate*(reward + self.discount_factor*best_future_q_value)

        self.q_table[state[1], state[0], action] = new_q_value

#reward system

def get_reward(current_state, next_state, death_eater_pos, cup_pos):
    goal_reward=100
    death_eater_penalty=-40
    move_reward=0.5
    step_reward=-1
    
    is_done=False
    reward=0
    #Reward logic
    if next_state==cup_pos:
        reward=goal_reward
        is_done=True 
    elif next_state==death_eater_pos:
        reward=death_eater_penalty
        is_done=True
    elif next_state == current_state:
        reward=-5        #Penalty for not moving
    elif(MAZE[next_state[1]][next_state[0]]==1):
        reward=-5        #Penalty for hitting the wall 
    else:
        reward += move_reward  # reward for taking a step    
    
    #Penalty to walk towards the Deatheater
    distance=(next_state[0]-death_eater_pos[0])**2 + (next_state[1]-death_eater_pos[1])**2
    if distance<=9:
        reward+=-20+(distance+1)
    elif distance >9 and distance<25:
        reward+=10
    elif distance>=25:
        reward=0
    
    #step penalty
    reward += step_reward
        
    return reward, is_done
    
#Generating random cell
def gen_random_cell(exclude=[]):  #exclude is a list of cells to be avoided apart from the wall cell
    while True:
        x=random.randint(0, GRID_WIDTH - 1)
        y=random.randint(0, GRID_HEIGHT - 1)
        if MAZE[y][x]==0 and (x,y) not in exclude:  #(x,y) are co-ordinates of the patches but not the cellss to be excluded 
            return (x, y)

#Traversal Mechanism of Death Eater
def bfs(start, goal):
    queue = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path + [(x, y)]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and MAZE[ny][nx] == 0:
                if (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(x, y)]))
                    visited.add((nx, ny))
    return []

#Function to take action
def finish_episode(Harry,harry_pos, death_eater_pos, cup_pos, current_episode, train=True):

    current_state=harry_pos
    is_done=False
    episode_reward=0
    episode_step=0
    path=[current_state]

    while not is_done:
        if episode_step >= 100:  # max steps
            break

        action = Harry.get_valid_actions(current_state, current_episode)

        #Harry moves
        nx = current_state[0] + actions[action][0]
        ny = current_state[1] + actions[action][1]

        #Check within grid and not hitting wall
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and MAZE[ny][nx] == 0:
            next_state = (nx, ny)
        else:
            next_state = current_state  # Harry stays in place if move is invalid

        #Death eater moves
        DE_path=bfs(death_eater_pos, harry_pos)
        if len(DE_path)>1:
            death_eater_pos=DE_path[1]

        #Reward logic
        reward, is_done = get_reward(current_state, next_state, death_eater_pos, cup_pos)
        
        episode_reward+=reward
        episode_step+=1

        if train == True:
            Harry.update_q_value(current_state, action, next_state, reward)

        current_state=next_state
    
    return episode_reward, episode_step, path

#Training the agent 
def train_Harry(Harry, harry_pos, death_eater_pos, cup_pos, num_episodes):

    episode_rewards=[]
    episode_steps=[]

    for episode in range(num_episodes):
        harry_pos = gen_random_cell([cup_pos])
        death_eater_pos = gen_random_cell([harry_pos,cup_pos])
        episode_reward, episode_step, path=finish_episode(Harry, harry_pos, death_eater_pos, cup_pos, episode)
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
    
    # Plotting the data after training is completed
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")

    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 100)
    plt.title('Steps per Episode')

    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")

    plt.tight_layout()
    plt.savefig("training_plot4c.png")


#Initiliing the positions of the characters
cup_pos=(7,5)                                                 #position of the cup
harry_pos=gen_random_cell([cup_pos])                          #position of harry
death_eater_pos=gen_random_cell([harry_pos,cup_pos])          #position of death eater

Harry=QlearningAgent()
train_Harry(Harry, harry_pos, death_eater_pos, cup_pos, num_episodes=20000)

with open("Qc4_table.pkl", "wb") as f:
        pickle.dump(Harry.q_table, f)

print("Qc4-table saved!")

def evaluate_Harry(Harry, cup_pos, required_streak=10, max_generations=10000):
    success_streak = 0
    max_streak = 0
    generation_count = 0
    reached_target = False
    generations_to_target = None

    while generation_count < max_generations:
        harry_pos = gen_random_cell([cup_pos])
        death_eater_pos = gen_random_cell([harry_pos, cup_pos])

        reward, steps, path = finish_episode(Harry, harry_pos, death_eater_pos, cup_pos, current_episode=0, train=False)
        generation_count += 1

        if path[-1] == cup_pos:
            success_streak += 1
            if success_streak == required_streak and not reached_target:
                generations_to_target = generation_count
                reached_target = True
        else:
            success_streak = 0

        max_streak = max(max_streak, success_streak)

    print(f"🎯 Generations to reach {required_streak} consecutive escapes: {generations_to_target if generations_to_target is not None else 'Not achieved'}")
    print(f"🏆 Maximum escape streak during evaluation: {max_streak}")


evaluate_Harry(Harry, cup_pos)