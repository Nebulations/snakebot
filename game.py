import pygame
import random

class SnakeGame():
    def __init__(self, render=False, fps: int = 5, grid: tuple[int] = (20, 16, 32, 4)):
        #                 up      right    down     left
        self.offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        self.GRID_WIDTH = grid[0]
        self.GRID_HEIGHT = grid[1]
        self.SQUARE_SIZE = grid[2] # pixels
        self.SPACING = grid[3]

        # Asthetics
        self.SNAKE_COLOR = (0, 255, 0)
        self.BACKGROUND_COLOR = (0, 0, 0)
        self.APPLE_COLOR = (255, 0, 0)

        self.render = render
        self.running = True
        self.frame_iteration = 0
        self.fps = fps

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.GRID_WIDTH*self.SQUARE_SIZE, self.GRID_HEIGHT*self.SQUARE_SIZE))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Snake")

        self.snake = {
            "head": [self.GRID_WIDTH//2, self.GRID_HEIGHT//2],
            # UP: 0, RIGHT: 1, BOTTOM: 2, LEFT: 3
            "direction": 1, # Equivilent of going right
            "length": 3
        }

        # Init grid to be empty
        self.grid = [[0 for y in range(self.GRID_HEIGHT)] for x in range(self.GRID_WIDTH)]
        # Put the snake's head on the grid.
        self.grid[self.GRID_WIDTH//2][self.GRID_HEIGHT//2] = 1

        self.generate_apple()

    def is_colliding(self, point: list | tuple | None = None) -> bool:
        if point is None:
            point = self.snake["head"]

        # print("Point:",point)

        x, y = point

        # Out of bounds on the horizontal (x) axis
        if x < 0 or x >= self.GRID_WIDTH:
            return True

        # Out of bounds on the vertical (y) axis
        if y < 0 or y >= self.GRID_HEIGHT:
            return True

        # Snake is colliding with itself
        # We do > 1 here because if the grid has 1
        # there, it's the snake's tail and thus
        # will move out of the way once the whole
        # snake updates.
        if self.grid[x][y] > 1:
            return True

        return False

    def step(self, action) -> tuple[float, float, bool]:
        reward, score, done = 0, self.snake["length"] - 3, False

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return 0, 0, True

        # 1. Update snake head location based off the given action
        directions = ["up", "right", "down", "left"]

        # Turning left
        if action == [1, 0, 0]:
            self.snake["direction"] = (self.snake["direction"] - 1) % 4
        # Turning right
        elif action == [0, 0, 1]:
            self.snake["direction"] = (self.snake["direction"] + 1) % 4
        elif action == [0, 1, 0]:
            pass

        # Update snake head accordingly
        match directions[self.snake["direction"]]:
            case "up": self.snake["head"][1]-=1
            case "left": self.snake["head"][0]-=1
            case "down": self.snake["head"][1]+=1
            case "right": self.snake["head"][0]+=1        

        # 2. Determine if the snake is colliding and should be dying
        if self.is_colliding():
            # -10 reward, score, and game ended.
            # Do -3 on length because snake starts at 3.
            return -10, score, True

        # 3. Snake took too long to eat an apple
        # self.frame_iteration+=1
        # if self.frame_iteration > 100 * self.snake["length"]:
        #     return -10, score, True

        # 3. Check if apple was ate
        if self.grid[self.snake["head"][0]][self.snake["head"][1]] == -1:
            self.snake["length"] += 1
            self.generate_apple()

            self.frame_iteration = 0
            reward, score, done = 10, score+1, False

        # 4. Tick through all snake tiles, and create new tile with the new head position.
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x][y] > 0:
                    self.grid[x][y] -= 1

        self.grid[self.snake["head"][0]][self.snake["head"][1]] = self.snake["length"]

        # 5. Render if needed
        if self.render:
            self.draw_snake()
            self.clock.tick(self.fps)

        # 6. Return the correct reward, score and if we're done.
        return reward, score, done

    # Danger (colliding left, in front, right)
    # Direction (up, right, down, left)
    # Apple relative to snake head (up, right, down, left)
    # Amount of space in 4 grids of the board.
    def get_state(self) -> list[int]:
        head_x, head_y = self.snake["head"]
        snake_direction = self.snake["direction"]

        # Danger: -1 = turn left, 0 = straight, 1 = turn right
        danger = []
        for turn in (-1, 0, 1):
            dx, dy = self.offsets[(snake_direction + turn) % 4]
            danger.append(int(self.is_colliding([head_x + dx, head_y + dy])))

        # Direction: one-hot, only the slot for the current direction is 1
        direction = [int(i == snake_direction) for i in range(4)]

        # Apple: y grows downward, so "up" means a smaller y
        apple_x, apple_y = self.apple
        apple = [
            int(apple_y < head_y),  # up
            int(apple_x > head_x),  # right
            int(apple_y > head_y),  # down
            int(apple_x < head_x),  # left
        ]

        # Grid areas (top left, top right, bottom left, bottom right)
        area_inputs = self.get_area_inputs()

        # up right down left
        rays = [0, 0, 0, 0]
        current = 0
        count = 0
        # Ray casts from the snake's head to the left, forwards, and right.
        # Pseudo-code (for when we're going up):
        # 1. Loop through the values on the left of the snake's head (head_x-1 -> 0)
        # 2. Right of snake head (head_x+1 -> width)
        # 3. In front of snake head (head_y-1 -> 0)
        # 4. Behind head (head_y+1 -> height)
        for x in range(head_x - 1, -1, -1):
            point = (x, head_y)
            if self.is_colliding(point):
                rays[current] = count
                count = 0
                current += 1
                break
            count += 1

        for x in range(head_x + 1, self.GRID_WIDTH):
            point = (x, head_y)
            if self.is_colliding(point):
                rays[current] = count
                count = 0
                current += 1
                break
            count += 1

        for y in range(head_y - 1, -1, -1):
            point = (head_x, y)
            if self.is_colliding(point):
                rays[current] = count
                count = 0
                current += 1
                break
            count += 1

        for y in range(head_y + 1, self.GRID_HEIGHT):
            point = (head_x, y)
            if self.is_colliding(point):
                rays[current] = count
                count = 0
                current += 1
                break
            count += 1

        # print("Calculated rays:", rays)

        return danger + direction + apple + area_inputs + rays

    def stop(self):
        pygame.quit()

    def reset(self):
        # (self, render=False, fps: int = 5, grid: tuple[int] = (20, 16, 32, 4)):
        self.__init__(render=self.render, fps=self.fps, grid=(self.GRID_WIDTH, self.GRID_HEIGHT, self.SQUARE_SIZE, self.SPACING))

    def generate_apple(self):
        while True:
            x, y = random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1)
            if self.grid[x][y] == 0:
                self.grid[x][y] = -1
                self.apple = [x, y]
                return

    def draw_snake(self):
        self.screen.fill(self.BACKGROUND_COLOR)

        # Render game
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                
                # == -1: Apple
                # == 0: Nothing to display
                # >= 1: Snake
                tile_type = self.grid[x][y]

                match tile_type:
                    # Apple
                    case -1:
                        pygame.draw.rect(self.screen, self.APPLE_COLOR, (x*self.SQUARE_SIZE+self.SPACING, y*self.SQUARE_SIZE+self.SPACING, self.SQUARE_SIZE-self.SPACING, self.SQUARE_SIZE-self.SPACING), 0)    
                    # Do nothing
                    case 0:
                        continue
                    # Snake
                    case _:
                        pygame.draw.rect(self.screen, self.SNAKE_COLOR, (x*self.SQUARE_SIZE+self.SPACING, y*self.SQUARE_SIZE+self.SPACING, self.SQUARE_SIZE-self.SPACING, self.SQUARE_SIZE-self.SPACING), 0)

        pygame.display.flip()

    def get_area_inputs(self) -> list[float]:
        hx, hy = self.snake["head"]
        d = self.snake["direction"]
        fx, fy = self.offsets[d]                  # forward
        rx, ry = self.offsets[(d + 1) % 4]        # right

        free = [0, 0, 0, 0]    # front-left, front-right, back-left, back-right
        total = [0, 0, 0, 0]

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                dx, dy = x - hx, y - hy
                ahead = dx * fx + dy * fy         # >0 in front of the head
                side = dx * rx + dy * ry          # >0 to the right of the head
                if ahead == 0 or side == 0:
                    continue                      # skip the cells on the head's own row/column
                q = (0 if ahead > 0 else 2) + (1 if side > 0 else 0)
                total[q] += 1
                if self.grid[x][y] <= 0:          # empty or apple
                    free[q] += 1

        return [f / t if t else 0 for f, t in zip(free, total)]
