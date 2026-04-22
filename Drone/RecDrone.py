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
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = self.x + dx, self.y + dy

            if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                if (nx, ny) not in self.known_map:
                    path = self.route((nx, ny))
                    if path and len(path) > 1:
                        self.x, self.y = path[1]
                        break

def generate_spiral(self):
    path = []
    x, y = self.x, self.y

    step_size = 1
    directions = [(0,1), (1,0), (0,-1), (-1,0)]  # R, D, L, U

    while len(path) < len(self.grid) * len(self.grid[0]):
        for i, (dx, dy) in enumerate(directions):
            for _ in range(step_size):
                x += dx
                y += dy

                if 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]):
                    path.append((x, y))

            # Increase step every 2 directions → spiral
            if i % 2 == 1:
                step_size += 1

    return path                    

def init_spiral(self, grid_size, step):
    self.spiral_path = []
    top, bottom, left, right = 0, grid_size - 1, 0, grid_size - 1

    while top <= bottom and left <= right:
        for y in range(left, right + 1, step):
            self.spiral_path.append((top, y))
        for x in range(top, bottom + 1, step):
            self.spiral_path.append((x, right))
        for y in range(right, left - 1, -step):
            self.spiral_path.append((bottom, y))
        for x in range(bottom, top - 1, -step):
            self.spiral_path.append((x, left))

        top += step
        bottom -= step
        left += step
        right -= step

    self.spiral_index = 0


def recon_step(drone, grid_size, sense_radius):
    old_pos = (drone.x, drone.y)

    refresh_known_cells(drone)
    sense_cells(drone, sense_radius)

    if not hasattr(drone, "spiral_path"):
        drone.init_spiral(grid_size, step=2 * sense_radius)

    if drone.spiral_index < len(drone.spiral_path):
        target = drone.spiral_path[drone.spiral_index]
        if (drone.x, drone.y) == target:
            drone.spiral_index += 1
        else:
            move_one_step(drone, target)
    else:
        target = None  # done

    return True, old_pos, (drone.x, drone.y), target    