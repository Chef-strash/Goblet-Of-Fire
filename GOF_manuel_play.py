import pygame
import random
from collections import deque

TILE_SIZE = 50
GRID_WIDTH = 15
GRID_HEIGHT = 10
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE

#Colors scheme in RGB
WHITE=(255, 255, 255)
BLACK=(0, 0, 0)
GRAY=(200, 200, 200)
BLUE=(50, 50, 255)  #color of harry
RED=(255, 50, 50)   #color of the death eater
GREEN=(50, 255, 50) #color of the walls
GOLD=(255, 215, 0)  #color of the cup

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

#Drwading the maze
def draw_maze(screen):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            color=WHITE if MAZE[y][x]==0 else GREEN
            pygame.draw.rect(screen, color, (x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, WHITE, (x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE),1)

#Generating random cell
def gen_random_cell(exclude=[]):  #exclude is a list of cells to be avoided apart from the wall cell
    while True:
        x=random.randint(0, GRID_WIDTH - 1)
        y=random.randint(0, GRID_HEIGHT - 1)
        if MAZE[y][x]==0 and (x,y) not in exclude:  #(x,y) are co-ordinates of the patches but not the cellss to be excluded 
            return x, y

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

#Initilizing the Game
pygame.init()
screen=pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Harry vs Death Eater")
font=pygame.font.SysFont(None, 48)

#Initiliing the positions of the characters
harry_pos=gen_random_cell()                           #position of harry
death_eater_pos=gen_random_cell([harry_pos])          #position of death eater
cup_pos=7,5 #position of the cup

clock=pygame.time.Clock()
game_over=False
win=False

#Game Loop (Infinite loop till WIN or LOSS)
running=True
while running:
    clock.tick(3)
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False

    if not game_over:
        # Harry Movement
        keys=pygame.key.get_pressed()    #get the key that is pressed
        dx, dy=0, 0
        if keys[pygame.K_LEFT]: dx=-1
        elif keys[pygame.K_RIGHT]: dx=1
        elif keys[pygame.K_UP]: dy=-1
        elif keys[pygame.K_DOWN]: dy=1

        new_x=harry_pos[0] + dx
        new_y=harry_pos[1] + dy
        if 0<=new_x<GRID_WIDTH and 0<=new_y<GRID_HEIGHT:
            if MAZE[new_y][new_x]==0:
                harry_pos=(new_x, new_y)

        #Check win condition
        if harry_pos==cup_pos:
            game_over=True
            win=True

        #Death Eater move (BFS)
        path=bfs(death_eater_pos, harry_pos)
        if path and len(path)>1:
            death_eater_pos=path[1]

        #Check lose condition
        if harry_pos==death_eater_pos:
            game_over=True
            win=False

    #Drawing maze
    screen.fill(BLACK)
    draw_maze(screen)

    #Draw Cup
    pygame.draw.rect(screen, GOLD, (cup_pos[0]*TILE_SIZE, cup_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))

    #Draw Characters
    pygame.draw.rect(screen, BLUE, (harry_pos[0]*TILE_SIZE, harry_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))
    pygame.draw.rect(screen, RED, (death_eater_pos[0]*TILE_SIZE, death_eater_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))

   # if game_over:
       # message = "Harry Wins!" if win else "Caught by Death Eater!"
       # text = font.render(message, True, GREEN if win else RED)
       # screen.blit(text, ((SCREEN_WIDTH - text.get_width()) // 2, SCREEN_HEIGHT // 2))
    
    pygame.display.flip()

pygame.quit()
