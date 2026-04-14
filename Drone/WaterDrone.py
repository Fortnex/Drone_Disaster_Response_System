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

    def nearest(self, targets):
        best = None
        best_dist = float('inf')

        for t in targets:
            dist = abs(self.x - t[0]) + abs(self.y - t[1])
            if dist < best_dist:
                best_dist = dist
                best = t

        return best

    def decide_target(self):
        fires = [pos for pos, val in self.known_map.items() if val == 1]
        waters = [pos for pos, val in self.known_map.items() if val == 3]
        chargers = [pos for pos, val in self.known_map.items() if val == 2]

        # Priority 1: battery
        if self.battery < 10 and chargers:
            return self.nearest(chargers)

        # Priority 2: water
        if self.water == 0 and waters:
            return self.nearest(waters)

        # Priority 3: fire
        if fires:
            return self.nearest(fires)

        return None

    def step(self, drones):
        # interact with environment
        self.extinguish()
        self.refill_water()

        target = self.decide_target()
        if not target:
            return

        path = self.route(target)
        if path and len(path) > 1:
            self.x, self.y = path[1]
            self.battery -= 1