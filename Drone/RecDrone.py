from Definition_Drone import Drone


class RecDrone(Drone):
    def __init__(self, id, battery, batterymax, x, y, grid):
        super().__init__(id, battery, batterymax, x, y, grid)
        self.known_map = {(self.x, self.y): self.grid[self.x][self.y]}

    def sense(self, radius=3):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                    self.known_map[(nx, ny)] = self.grid[nx][ny]

    def in_range(self, other, r=4):
        return abs(self.x - other.x) + abs(self.y - other.y) <= r

    def broadcast(self, drones):
        useful = {k: v for k, v in self.known_map.items() if v != 0}

        for d in drones:
            if d != self and self.in_range(d):
                if hasattr(d, "known_map"):
                    d.known_map.update(useful)

    def step(self, drones):
        # Sense environment
        self.sense()

        # Share information
        self.broadcast(drones)

        # Simple exploration: move to nearby unknown cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = self.x + dx, self.y + dy

            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                if (nx, ny) not in self.known_map:
                    path = self.route((nx, ny))
                    if path and len(path) > 1:
                        self.x, self.y = path[1]
                        break
