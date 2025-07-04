import pygame
import math
from queue import PriorityQueue
pygame.init()
Width = 800
win = pygame.display.set_mode((Width,Width))
pygame.display.set_caption("A * SEARCH")
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Node:
    def __init__(self,row,col,width,total_rows):
        self.row=row
        self.col = col
        self.x = row*width
        self.y= col * width
        self.colors = WHITE
        self.neighbour = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row,self.col
    
    def is_closed(self):
        return self.colors == RED
    
    def is_open(self):
        return self.colors == GREEN
    
    def is_barrier(self):
        return self.colors == BLACK
    def is_start(self):
        return self.colors == ORANGE
    def is_end(self):
        return self.colors == TURQUOISE
    def reset(self):
        self.colors = WHITE
    
    def make_closed(self):
        self.colors = RED
    def make_open(self):
        self.colors = GREEN
    def make_barrier(self):
        self.colors = BLACK
    def make_start(self):
        self.colors=ORANGE
    def make_end(self):
        self.colors=TURQUOISE
    def make_path(self):
        self.colors=PURPLE

    def draw(self,win):
        pygame.draw.rect(win,self.colors,(self.x,self.y,self.width,self.width))

    def update_neighbours(self,grid):
        self.neighbour=[]
        if self.row<self.total_rows-1 and not grid[self.row+1][self.col].is_barrier():
            self.neighbour.append(grid[self.row+1][self.col])

        if self.row>0 and not grid[self.row-1][self.col].is_barrier():
            self.neighbour.append(grid[self.row-1][self.col])

        if self.col<self.total_rows-1 and not grid[self.row][self.col+1].is_barrier():
            self.neighbour.append(grid[self.row][self.col+1])

        if self.col>0 and not grid[self.row][self.col-1].is_barrier():
            self.neighbour.append(grid[self.row][self.col-1])


    def __lt__(self,other):
        return False
    
def h(p1,p2):
    x1,y1 = p1
    x2 , y2 = p2
    return abs(x2-x1) + abs(y2-y1)

def make_grid(rows,width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i,j,gap,rows)
            grid[i].append(node)
    return grid

def draw_grid(win,rows,width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win,GREY,(0,i*gap),(width,i*gap))
        for j in range(rows):
            pygame.draw.line(win,GREY,(j*gap,0),(j*gap,width))

def draw(win,grid,rows,width):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win) 
    draw_grid(win,rows,width)
    pygame.display.update()

def on_clicked_pos(pos,rows,width):
    gap = width // rows
    x,y = pos
    row = x // gap
    col = y // gap
    return row,col

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()


def algorithm(draw,grid,start,end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0,count,start))
    came_from = {}
    g_score = {node:float("inf") for row in grid for node in row}
    g_score[start]=0
    f_score = {node:float("inf") for row in grid for node in row}
    f_score[start]=h(start.get_pos(),end.get_pos())

    open_set_hash = {start} #Pririty q me konse elemnts hai woh pta krne ke liye
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbour in current.neighbour:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()
        if current != start:
            current.make_closed()

    return False



def main(win,width):
    ROWS = 50
    grid = make_grid(ROWS,width)

    start = None
    end = None

    run = True
    started = False

    while run:
        draw(win,grid,ROWS,width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue
            if pygame.mouse.get_pressed()[0]: #Left mouse btn
                pos = pygame.mouse.get_pos()
                row,col = on_clicked_pos(pos,ROWS,width)
                node = grid[row][col]
                if not start and node!=end:
                    start = node
                    start.make_start()

                elif not end and node !=start:
                    end = node
                    end.make_end()

                elif node != end and node != start:
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]: #Right click
                pos = pygame.mouse.get_pos()
                row,col = on_clicked_pos(pos,ROWS,width)
                node = grid[row][col]
                node.reset()
                if node==start:
                    start = None
                if node==end:
                    end=None

            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)
                    algorithm(lambda:draw(win,grid,ROWS,Width),grid,start,end)
    pygame.quit()

main(win,Width)








