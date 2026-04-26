import random


class MakeGrid:
    def __init__(self, fire, water, gridSize, obstacle):
        self.num_fire = fire
        self.num_water = water
        self.grid_size = gridSize
        self.num_obstacle = obstacle
        self.grid = [[0 for _ in range(gridSize)] for _ in range(gridSize)]
        self.place_random(self.num_fire, 1)
        self.place_random(self.num_obstacle, 2)
        self.place_random(self.num_water, 3)

    def generate(self):
        return self.grid

    def place_random(self, num, type_id):
        placed = 0
        while placed < num:
            rx = random.randint(0, self.grid_size - 1)
            ry = random.randint(0, self.grid_size - 1)
            if self.grid[rx][ry] == 0:
                self.grid[rx][ry] = type_id
                placed += 1
