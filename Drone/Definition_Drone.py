import random
import math
class Drone:
    def __init__(self,id,battery,batterymax,water,watermax,x,y,typeofDrone):
        self.id = id
        self.battery = battery
        self.watermax = watermax
        self.batterymax = batterymax
        self.water = water
        self.x = x
        self.y = y
        self.typeofDrone = typeofDrone
    def isWaterDrone(self):
        if self.typeofDrone == 1:
            return True
        return False
    def battery_usage(self):
        self.battery-=1
    def can_use_battery(self):
        if self.batter!=0:
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
    def movement(self,x,y):
        self.x+=x
        self.y+=y

class MakeGrid:
    def __init__(self, fire, water, gridSize, charging):
        self.num_fire = fire
        self.num_water = water
        self.grid_size = gridSize
        self.num_charging = charging
        self.grid = [[0 for _ in range(gridSize)] for _ in range(gridSize)]

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