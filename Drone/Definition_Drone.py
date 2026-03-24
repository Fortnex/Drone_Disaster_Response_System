import random
import math
import heapq
#1 1 = Fire 2 = charging  3 = water  "
class Drone:
    def __init__(self,id,battery,batterymax,water,watermax,x,y,typeofDrone,grid):
        self.id = id
        self.battery = battery
        self.watermax = watermax
        self.batterymax = batterymax
        self.water = water
        self.x = x
        self.y = y
        self.typeofDrone = typeofDrone
        self.grid = grid
    def isWaterDrone(self):
        if self.typeofDrone == 1:
            return True
        return False
    def battery_usage(self):
        self.battery-=1
    def can_use_battery(self):
        if self.battery!=0:
            return True
        else:
            return False
    def refill(self,a):
        if(self.isWaterDrone()):
            self.water = min(self.watermax,self.water+a)
    def can_use_water(self,a):
        if self.water>=a:
            return True
        else:
            return False
    def water_usage(self,a):
        self.water = self.water=self.water-a

    def moveX_forward(self):
        self.x+=1

    def moveX_backward(self):
        self.x-=1

    def moveY_forward(self):
        self.y+=1

    def moveY_backward(self):
        self.y-=1
    
    def route(self,source, dest):
        grid = self.grid
        rows = len(grid)
        cols = len(grid[0])
        
        # Priority Queue: (priority, x, y, path)
        # Priority is g (cost so far) + h (estimated cost to goal)
        pq = [(0, source[0], source[1], [source])]
        visited = set()

        while pq:
            (cost, x, y, path) = heapq.heappop(pq)

            if (x, y) == dest:
                return path

            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Explore North, South, East, West
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # 1. Check if within map boundaries
                if 0 <= nx < rows and 0 <= ny < cols:
                    # 2. STRICT MOVE RULE: Must be 0 OR the final destination
                    if grid[nx][ny] == 0 or (nx, ny) == dest:
                        if (nx, ny) not in visited:
                            new_path = path + [(nx, ny)]
                            # Manhattan distance heuristic
                            h = abs(nx - dest[0]) + abs(ny - dest[1])
                            g = len(new_path)
                            heapq.heappush(pq, (g + h, nx, ny, new_path))
                        
        return None # Path is blocked by obstacles

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
    def findDrone(self,id):
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == id:
                    return (r, c)
        return None
    

    