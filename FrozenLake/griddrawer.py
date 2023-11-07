import pygame
import time

class OUR_RENDERER():
    def __init__(self, environment, size):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)

        self.WIDTH = 40
        self.HEIGHT = 40

        self.MARGIN = 5
        self.SIZE = size

        grid = []

        for row in range(self.SIZE):
            grid.append([])
            for column in range(self.SIZE):
                grid[row].append(0)
     

        _start = environment["start"]
        _fin = environment["fin"]
        _obstacle = environment["obstacle"]

        for cord in _start:
            grid[cord[0]][cord[1]] = 2
        for cord in _fin:
            grid[cord[0]][cord[1]] = 3
        for cord in _obstacle:
            grid[cord[0]][cord[1]] = 4

        self.grid = grid
        self.WINDOW_SIZE = [self.WIDTH * self.SIZE + self.MARGIN*(self.SIZE + 1), 
        self.WIDTH * self.SIZE + self.MARGIN*(self.SIZE + 1)]

        pygame.init()
     
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
     
        pygame.display.set_caption("Frozenlake")
     
        self.done = False
     
        self.clock = pygame.time.Clock()
 
    def renderer(self, path_r):
        time.sleep(1)
        state_row, state_col = 0, 0
        N_iter = 0
        while not self.done:
            cur_grid = self.grid[:]
            if state_row == self.SIZE - 1 and state_col == self.SIZE -1:
                self.done = True
                time.sleep(1)
                break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True

            if path_r[state_row, state_col] == 'left':
                state_col -= 1
            elif path_r[state_row, state_col] == 'right':
                state_col += 1
            elif path_r[state_row, state_col] == 'up':
                state_row -= 1
            elif path_r[state_row, state_col] == 'down':
                state_row += 1

            if state_row < 0 or state_col < 0 or state_row > self.SIZE - 1 or state_col > self.SIZE - 1:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("ERROR: PLAYER OUT OF THE WORLD!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                time.sleep(1)
                self.done = True
            elif cur_grid[state_row][state_col] == 4:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("ERROR: PLAYER HIT AN OBSTACLE !")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                time.sleep(1)
                self.done = True
            else:
                cur_grid[state_row][state_col] = 1
                         
            self.screen.fill(self.BLACK)

            for row in range(self.SIZE):
                for column in range(self.SIZE):

                    if cur_grid[row][column] == 2:
                        color = self.YELLOW
                    elif cur_grid[row][column] == 3:
                        color = self.BLUE
                    elif cur_grid[row][column] == 4:
                        color = self.RED
                    elif cur_grid[row][column] == 1:
                        color = self.GREEN
                    else:
                        color = self.WHITE

                    pygame.draw.rect(self.screen,
                                     color,
                                     [(self.MARGIN + self.WIDTH) * column + self.MARGIN,
                                      (self.MARGIN + self.HEIGHT) * row + self.MARGIN,
                                      self.WIDTH,
                                      self.HEIGHT])
         
            N_iter += 1
            time.sleep(.5)
            self.clock.tick(60)
         
            pygame.display.flip()
         
        pygame.quit()

