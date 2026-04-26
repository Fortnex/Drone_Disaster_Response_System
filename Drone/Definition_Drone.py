
import heapq


class Drone:
    def __init__(self, id, battery, batterymax, x, y, grid):
        self.id = id
        self.battery = battery        
        self.batterymax = batterymax        
        self.x = x
        self.y = y        
        self.grid = grid
        
    def battery_usage(self):
        self.battery -= 1

    def can_use_battery(self):
        return self.battery != 0
    
    def route(self, dest):
        grid = self.grid
        rows, cols = len(grid), len(grid[0])

        start = (self.x, self.y)

        # Priority queue: (f, x, y)
        pq = []
        heapq.heappush(pq, (0, start[0], start[1]))

        came_from = {}
        g_cost = {start: 0}

        while pq:
            _, x, y = heapq.heappop(pq)

            if (x, y) == dest:
                path = []
                curr = dest
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                return path

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if 0 <= nx < rows and 0 <= ny < cols:
                    if grid[nx][ny] == 0 or (nx, ny) == dest:

                        new_g = g_cost[(x, y)] + 1

                        if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                            g_cost[(nx, ny)] = new_g

                            h = abs(nx - dest[0]) + abs(ny - dest[1])
                            f = new_g + h

                            heapq.heappush(pq, (f, nx, ny))
                            came_from[(nx, ny)] = (x, y)

        return None
