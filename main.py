import argparse
import os
import random
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_DIR = os.path.join(BASE_DIR, "Drone")
GRID_DIR = os.path.join(BASE_DIR, "Grid")

if DRONE_DIR not in sys.path:
    sys.path.append(DRONE_DIR)
if GRID_DIR not in sys.path:
    sys.path.append(GRID_DIR)

from MakeGrid import MakeGrid
from RecDrone import RecDrone
from WaterDrone import WaterDrone


def fire_count(grid):
    return sum(1 for row in grid for cell in row if cell == 1)


class MissionSimulator:
    def __init__(self, grid_size=10, fire=12, water=6, charging=4, water_drones=3, seed=7):
        random.seed(seed)
        self.world = MakeGrid(fire=fire, water=water, gridSize=grid_size, charging=charging)
        self.grid = self.world.generate()
        self.grid_size = grid_size
        self.recon_radius = 2
        self.discovery_announced = {"fire": False, "water": False}

        occupied = set()
        rec_pos = self._pick_empty_cell(occupied)
        self.rec_drone = RecDrone(90, battery=250, batterymax=250, x=rec_pos[0], y=rec_pos[1], grid=self.grid)

        self.water_drones = []
        for idx in range(water_drones):
            pos = self._pick_empty_cell(occupied)
            d = WaterDrone(
                id=10 + idx,
                battery=220,
                batterymax=220,
                water=0,
                watermax=4,
                x=pos[0],
                y=pos[1],
                grid=self.grid,
            )
            self.water_drones.append(d)

        print("Mission initialized")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Initial fires: {fire_count(self.grid)}")
        print(f"Recon drone start: {(self.rec_drone.x, self.rec_drone.y)}")
        for d in self.water_drones:
            print(f"Water drone {d.id} start: {(d.x, d.y)}")

    def _pick_empty_cell(self, occupied, near=None, max_dist=4):
        local_candidates = []
        global_candidates = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] != 0 or (r, c) in occupied:
                    continue
                global_candidates.append((r, c))
                if near is not None and abs(near[0] - r) + abs(near[1] - c) <= max_dist:
                    local_candidates.append((r, c))

        if near is not None and local_candidates:
            chosen = random.choice(local_candidates)
        else:
            chosen = random.choice(global_candidates)

        occupied.add(chosen)
        return chosen

    def _nearest_unknown(self):
        best = None
        best_dist = float("inf")
        sx, sy = self.rec_drone.x, self.rec_drone.y

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in self.rec_drone.known_map:
                    continue
                dist = abs(sx - r) + abs(sy - c)
                if dist < best_dist:
                    best_dist = dist
                    best = (r, c)
        return best

    def _known_positions(self, drone, cell_value):
        return [pos for pos, value in drone.known_map.items() if value == cell_value]

    def recon_step(self, step_no, occupied_positions, reserved_targets):
        prev_pos = (self.rec_drone.x, self.rec_drone.y)
        previous_sizes = {d.id: len(d.known_map) for d in self.water_drones}
        had_fire_intel = {d.id: any(v == 1 for v in d.known_map.values()) for d in self.water_drones}

        for pos in list(self.rec_drone.known_map.keys()):
            self.rec_drone.known_map[pos] = self.grid[pos[0]][pos[1]]

        self.rec_drone.sense(radius=self.recon_radius)
        known_fires = self._known_positions(self.rec_drone, 1)
        known_waters = self._known_positions(self.rec_drone, 3)

        if known_waters and not self.discovery_announced["water"]:
            self.discovery_announced["water"] = True
            print(f"[Step {step_no}] RecDrone discovered water at {known_waters[0]}")
        if known_fires and not self.discovery_announced["fire"]:
            self.discovery_announced["fire"] = True
            print(f"[Step {step_no}] RecDrone discovered fire at {known_fires[0]}")

        target = self._nearest_unknown()
        if target and self.rec_drone.can_use_battery():
            path = self.rec_drone.route(target)
            if path and len(path) > 1:
                candidate = path[1]
                occupied_by_others = occupied_positions - {prev_pos}
                if candidate in occupied_by_others or candidate in reserved_targets:
                    print(
                        f"[Step {step_no}] RecDrone held at {prev_pos} "
                        f"(collision avoidance on {candidate})"
                    )
                else:
                    self.rec_drone.x, self.rec_drone.y = candidate
                    self.rec_drone.battery_usage()
                    occupied_positions.remove(prev_pos)
                    occupied_positions.add(candidate)
                    reserved_targets.add(candidate)
                    print(f"[Step {step_no}] RecDrone moved {prev_pos} -> {(self.rec_drone.x, self.rec_drone.y)}")

        self.rec_drone.broadcast(self.water_drones)
        for d in self.water_drones:
            shared_cells = len(d.known_map) - previous_sizes[d.id]
            if shared_cells > 0:
                print(f"[Step {step_no}] Gossip: RecDrone shared {shared_cells} cells with WaterDrone {d.id}")
            has_fire_now = any(v == 1 for v in d.known_map.values())
            if has_fire_now and not had_fire_intel[d.id]:
                print(f"[Step {step_no}] Gossip alert to WaterDrone {d.id}: fire location received")

    def water_step(self, step_no, occupied_positions, reserved_targets):
        for d in self.water_drones:
            for pos in list(d.known_map.keys()):
                d.known_map[pos] = self.grid[pos[0]][pos[1]]

            knows_fire = any(v == 1 for v in d.known_map.values())
            if not knows_fire:
                print(f"[Step {step_no}] WaterDrone {d.id} waiting for fire intel")
                continue

            old_pos = (d.x, d.y)
            old_battery = d.battery
            old_water = d.water
            fires_before = fire_count(self.grid)

            d.extinguish()
            d.refill_water()
            target = d.decide_target()
            if target:
                path = d.route(target)
                if path and len(path) > 1:
                    candidate = path[1]
                    occupied_by_others = occupied_positions - {old_pos}
                    if candidate in occupied_by_others or candidate in reserved_targets:
                        print(
                            f"[Step {step_no}] WaterDrone {d.id} held at {old_pos} "
                            f"(collision avoidance on {candidate})"
                        )
                    else:
                        d.x, d.y = candidate
                        d.battery -= 1
                        occupied_positions.remove(old_pos)
                        occupied_positions.add(candidate)
                        reserved_targets.add(candidate)

            fires_after = fire_count(self.grid)
            if (d.x, d.y) != old_pos:
                print(
                    f"[Step {step_no}] WaterDrone {d.id} moved {old_pos} -> {(d.x, d.y)} "
                    f"(water={d.water}, battery={d.battery})"
                )
            if old_water == 0 and d.water == d.watermax:
                print(f"[Step {step_no}] WaterDrone {d.id} refilled at water source")
            if fires_after < fires_before:
                print(f"[Step {step_no}] WaterDrone {d.id} extinguished a fire")
            if (d.x, d.y) == old_pos and old_battery == d.battery and fires_after == fires_before:
                print(f"[Step {step_no}] WaterDrone {d.id} holding position")

    def render(self, ax, cmap, norm, radius_cmap):
        import numpy as np

        frame = [row[:] for row in self.grid]
        for d in self.water_drones:
            frame[d.x][d.y] = 4
        frame[self.rec_drone.x][self.rec_drone.y] = 5

        ax.clear()
        ax.imshow(np.array(frame), cmap=cmap, norm=norm)

        # Translucent overlay to show current sensing range of the recon drone.
        radius_mask = np.zeros((self.grid_size, self.grid_size))
        for dx in range(-self.recon_radius, self.recon_radius + 1):
            for dy in range(-self.recon_radius, self.recon_radius + 1):
                nx = self.rec_drone.x + dx
                ny = self.rec_drone.y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    radius_mask[nx][ny] = 1
        masked_radius = np.ma.masked_where(radius_mask == 0, radius_mask)
        ax.imshow(masked_radius, cmap=radius_cmap, alpha=0.30, vmin=1, vmax=1)

        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8)
        ax.set_title(
            f"Drone Mission | Fires left: {fire_count(self.grid)} | "
            f"Rec battery: {self.rec_drone.battery} | Sense radius: {self.recon_radius}"
        )

    def run(self, steps=120, delay=0.15, no_ui=False):
        if not no_ui:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(7, 7))
            cmap = mcolors.ListedColormap(
                [
                    "#2E7D32",  # 0 empty / forest
                    "#D32F2F",  # 1 fire
                    "#FFD54F",  # 2 charging
                    "#1976D2",  # 3 water source
                    "#212121",  # 4 water drone
                    "#8E24AA",  # 5 recon drone (distinct color)
                ]
            )
            bounds = [0, 1, 2, 3, 4, 5, 6]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            radius_cmap = mcolors.ListedColormap(["#81D4FA"])
            print("UI enabled: live mission window launched")
        else:
            plt = None
            ax = None
            cmap = None
            norm = None
            radius_cmap = None

        for step_no in range(1, steps + 1):
            print(f"\n=== Step {step_no} ===")
            occupied_positions = {(self.rec_drone.x, self.rec_drone.y)}
            occupied_positions.update((d.x, d.y) for d in self.water_drones)
            reserved_targets = set()

            self.recon_step(step_no, occupied_positions, reserved_targets)
            self.water_step(step_no, occupied_positions, reserved_targets)
            print(f"[Step {step_no}] Remaining fires: {fire_count(self.grid)}")

            if not no_ui:
                self.render(ax, cmap, norm, radius_cmap)
                plt.draw()
                plt.pause(delay)
            else:
                time.sleep(min(delay, 0.001))

            if fire_count(self.grid) == 0:
                print(f"\nMission success: all fires extinguished in {step_no} steps")
                break
        else:
            print("\nMission ended: step limit reached")

        if not no_ui:
            plt.ioff()
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Drone disaster response simulation")
    parser.add_argument("--steps", type=int, default=120, help="Maximum mission steps")
    parser.add_argument("--delay", type=float, default=0.45, help="UI frame delay in seconds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible map")
    parser.add_argument("--no-ui", action="store_true", help="Run without matplotlib window")
    args = parser.parse_args()

    sim = MissionSimulator(seed=args.seed)
    sim.run(steps=args.steps, delay=args.delay, no_ui=args.no_ui)


if __name__ == "__main__":
    main()
