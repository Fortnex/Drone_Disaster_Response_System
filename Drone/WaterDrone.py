from Definition_Drone import Drone


class WaterDrone(Drone):
    def __init__(self, id, battery, batterymax, water, watermax, x, y, grid):
        super().__init__(id, battery, batterymax, x, y, grid)
        self.water = water
        self.watermax = watermax
        self.known_map = {}

    def refill_water(self):
        if self.grid[self.x][self.y] == 3:
            self.water = self.watermax

    def extinguish(self):
        if self.grid[self.x][self.y] == 1 and self.water > 0:
            self.water -= 1
            self.grid[self.x][self.y] = 0
