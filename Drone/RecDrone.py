from Definition_Drone import Drone


class RecDrone(Drone):
    def __init__(self, id, battery, batterymax, x, y, grid):
        super().__init__(id, battery, batterymax, x, y, grid)
        self.known_map = {(self.x, self.y): self.grid[self.x][self.y]}
