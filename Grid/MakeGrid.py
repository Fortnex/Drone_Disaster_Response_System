import random


class MakeGrid:
    def __init__(self, fire, water, gridSize, charging):
        self.num_fire = fire
        self.num_water = water
        self.grid_size = gridSize
        self.num_charging = charging
        self.grid = [[0 for _ in range(gridSize)] for _ in range(gridSize)]
        self.place_random(self.num_fire, 1)      # 1 = Fire
        self.place_random(self.num_charging, 2)  
        self.place_random(self.num_water, 3)     # 3 = Water

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

    def print_Grid(self):
        for i in self.grid:
            print(str(i)+"\n")
        print("\nGlossary : \n 0 = Forest/Nothing \n 1 = Fire \n 2 = charging \n 3 = water  ")
    
    def place(self,x,y,value):
        self.grid[x][y]=value

    def findWater(self):
        return [(r, c) for r, row in enumerate(self.grid) 
               for c, val in enumerate(row) 
               if val == 3]
    def findFire(self):
        return [(r, c) for r, row in enumerate(self.grid) 
               for c, val in enumerate(row) 
               if val == 1]
    def findDrone(self,id):
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == id:
                    return (r, c)
        return None
    