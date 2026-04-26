import argparse
import os
import random
import sys

from mpi4py import MPI


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


TAG_GOSSIP = 100
TAG_FIRE_UPDATE = 101
FALLBACK_AFTER_SILENT_CYCLES = 2
FIRE_UPDATE_RELAY_CYCLES = 6
DEFAULT_MINI_LOG = "mpi_mini_debug.log"


COMMUNICATION_COLORS = {
    "recon-to-water": "#1565C0",
    "water-to-water": "#F57C00",
    "fire-update": "#D32F2F",
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def fire_count(grid):
    return sum(1 for row in grid for cell in row if cell == 1)


def initial_fire_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 1]


def water_fire_status_snapshot(grid, fire_cells):
    return {cell: grid[cell[0]][cell[1]] for cell in fire_cells}


def reconcile_fire_status_from_waters(grid, fire_cells, snapshots):
    changed = 0
    for row, col in fire_cells:
        if grid[row][col] != 1:
            continue
        for snapshot in snapshots:
            if not snapshot:
                continue
            if snapshot.get((row, col)) == 0:
                grid[row][col] = 0
                changed += 1
                break
    return changed


def build_world(grid_size, fires, waters, charging, seed):
    random.seed(seed)
    world = MakeGrid(fire=fires, water=waters, gridSize=grid_size, charging=charging)
    return world.generate()


def recon_partition_bounds(grid_size, recon_count, role_index):
    recon_count = max(1, recon_count)
    role_index = max(0, min(role_index, recon_count - 1))
    partitions = split_recon_bounds((0, grid_size - 1, 0, grid_size - 1), recon_count)
    return partitions[role_index]


def split_recon_bounds(bounds, drone_count):
    if drone_count <= 1:
        return [bounds]

    top, bottom, left, right = bounds
    height = bottom - top + 1
    width = right - left + 1
    first_count = (drone_count + 1) // 2
    second_count = drone_count - first_count

    if width >= height and width > 1:
        first_width = proportional_split_size(width, first_count, drone_count)
        left_bounds = (top, bottom, left, left + first_width - 1)
        right_bounds = (top, bottom, left + first_width, right)
        return split_recon_bounds(left_bounds, first_count) + split_recon_bounds(right_bounds, second_count)

    if height > 1:
        first_height = proportional_split_size(height, first_count, drone_count)
        top_bounds = (top, top + first_height - 1, left, right)
        bottom_bounds = (top + first_height, bottom, left, right)
        return split_recon_bounds(top_bounds, first_count) + split_recon_bounds(bottom_bounds, second_count)

    return [bounds for _ in range(drone_count)]


def proportional_split_size(length, first_count, total_count):
    split_size = round(length * first_count / total_count)
    return max(1, min(length - 1, split_size))


def empty_cells_in_bounds(grid, bounds):
    top, bottom, left, right = bounds
    return [
        (row, col)
        for row in range(top, bottom + 1)
        for col in range(left, right + 1)
        if grid[row][col] == 0
    ]


def pick_start_positions(grid, count, seed, recon_count=0):
    rng = random.Random(seed + 1000)
    grid_size = len(grid)
    empty_cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == 0
    ]
    rng.shuffle(empty_cells)

    if len(empty_cells) < count:
        raise ValueError("Not enough empty cells to place all drones")

    positions = []
    used = set()

    for role_index in range(min(recon_count, count)):
        bounds = recon_partition_bounds(grid_size, recon_count, role_index)
        partition_cells = [cell for cell in empty_cells_in_bounds(grid, bounds) if cell not in used]
        if not partition_cells:
            partition_cells = [cell for cell in empty_cells if cell not in used]
        if not partition_cells:
            raise ValueError("Not enough empty cells to place recon drones")

        chosen = min(partition_cells, key=lambda cell: (cell[0], cell[1]))
        positions.append(chosen)
        used.add(chosen)

    for cell in empty_cells:
        if len(positions) >= count:
            break
        if cell not in used:
            positions.append(cell)
            used.add(cell)

    return positions


def make_drone(rank, role, role_index, start, grid):
    if role == "recon":
        return RecDrone(
            id=90 + role_index,
            battery=2500,
            batterymax=25000,
            x=start[0],
            y=start[1],
            grid=grid,
        )

    drone = WaterDrone(
        id=10 + role_index,
        battery=2000,
        batterymax=2000,
        water=0,
        watermax=4,
        x=start[0],
        y=start[1],
        grid=grid,
    )
    drone.known_map[(drone.x, drone.y)] = grid[drone.x][drone.y]
    return drone


def known_cells_payload(known_map):
    return [
        (pos[0], pos[1], value)
        for pos, value in known_map.items()
        if value != 0
    ]


def merge_cells(known_map, cells):
    changed = 0
    for row, col, value in cells:
        pos = (row, col)
        if known_map.get(pos) != value:
            changed += 1
        known_map[pos] = value
    return changed


def refresh_known_cells(drone):
    for pos in list(drone.known_map.keys()):
        drone.known_map[pos] = drone.grid[pos[0]][pos[1]]


def sense_cells(drone, radius):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx = drone.x + dx
            ny = drone.y + dy
            if 0 <= nx < len(drone.grid) and 0 <= ny < len(drone.grid[0]):
                drone.known_map[(nx, ny)] = drone.grid[nx][ny]


def drain_messages(
    comm,
    drone,
    grid,
    seen_messages,
    seen_fire_updates,
    pending_fire_updates,
    rank,
    step,
):
    status = MPI.Status()
    received = {"gossip": 0, "fire_update": 0, "changed_cells": 0}

    while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_GOSSIP, status=status):
        source = status.Get_source()
        message = comm.recv(source=source, tag=TAG_GOSSIP)
        message_id = message["message_id"]
        if message_id in seen_messages:
            continue

        seen_messages.add(message_id)
        changed = merge_cells(drone.known_map, message["cells"])
        received["gossip"] += 1
        received["changed_cells"] += changed

        print(
            f"[Step {step}] Rank {rank} received gossip from rank {source} "
            f"({message['sender_role']}): {changed} new/changed cells",
            flush=True,
        )

    while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_FIRE_UPDATE, status=status):
        source = status.Get_source()
        message = comm.recv(source=source, tag=TAG_FIRE_UPDATE)
        update_id = message.get("update_id", f"legacy:{source}:{step}:{message['location']}")
        if update_id in seen_fire_updates:
            continue

        seen_fire_updates.add(update_id)
        row, col = message["location"]
        grid[row][col] = 0
        drone.known_map[(row, col)] = 0
        received["fire_update"] += 1
        pending_fire_updates.append(
            {
                "update_id": update_id,
                "location": (row, col),
                "step": message.get("step", step),
                "source_rank": message.get("source_rank", source),
                "last_hop": rank,
                "retries_left": message.get("retries_left", FIRE_UPDATE_RELAY_CYCLES),
                "delivered_to": [],
            }
        )

        print(
            f"[Step {step}] Rank {rank} received fire update from rank {source}: "
            f"{(row, col)} is extinguished",
            flush=True,
        )

    return received


def send_gossip(
    comm,
    rank,
    role,
    drone,
    peer_states,
    target_roles,
    radio_range,
    step,
    reason,
    events=None,
):
    cells = known_cells_payload(drone.known_map)
    if not cells:
        return 0

    requests = []
    sent = 0
    sender_pos = (drone.x, drone.y)

    for peer in peer_states:
        if peer["rank"] == rank or peer["role"] not in target_roles:
            continue
        if manhattan(sender_pos, peer["position"]) > radio_range:
            continue

        payload = {
            "type": "gossip",
            "message_id": f"{rank}:{step}:{reason}:{peer['rank']}",
            "sender_rank": rank,
            "sender_id": drone.id,
            "sender_role": role,
            "sender_position": sender_pos,
            "step": step,
            "cells": cells,
        }
        requests.append(comm.isend(payload, dest=peer["rank"], tag=TAG_GOSSIP))
        sent += 1

        if events is not None:
            events.append(
                {
                    "type": "communication",
                    "kind": reason,
                    "from_rank": rank,
                    "to_rank": peer["rank"],
                    "from_role": role,
                    "to_role": peer["role"],
                    "from": sender_pos,
                    "to": peer["position"],
                    "cell_count": len(cells),
                }
            )

    if requests:
        MPI.Request.Waitall(requests)

    return sent


def broadcast_fire_update(
    comm,
    rank,
    location,
    step,
    peer_states,
    radio_range,
    seen_fire_updates,
    events=None,
):
    update_id = f"fire:{rank}:{step}:{location[0]}:{location[1]}"
    seen_fire_updates.add(update_id)
    payload = {
        "type": "fire_update",
        "update_id": update_id,
        "sender_rank": rank,
        "source_rank": rank,
        "location": location,
        "step": step,
        "last_hop": rank,
        "retries_left": FIRE_UPDATE_RELAY_CYCLES,
    }
    requests = []
    sent = 0

    for peer in peer_states:
        if peer["rank"] == rank:
            continue
        if manhattan(location, peer["position"]) > radio_range:
            continue
        requests.append(comm.isend(payload, dest=peer["rank"], tag=TAG_FIRE_UPDATE))
        sent += 1

    if requests:
        MPI.Request.Waitall(requests)

    if events is not None:
        for peer in peer_states:
            if peer["rank"] == rank:
                continue
            if manhattan(location, peer["position"]) > radio_range:
                continue
            events.append(
                {
                    "type": "communication",
                    "kind": "fire-update",
                    "from_rank": rank,
                    "to_rank": peer["rank"],
                    "from_role": "water",
                    "to_role": peer["role"],
                    "from": location,
                    "to": peer["position"],
                    "cell_count": 1,
                }
            )

    return sent, payload


def forward_fire_updates(
    comm,
    rank,
    peer_states,
    radio_range,
    pending_fire_updates,
    events=None,
):
    if not pending_fire_updates:
        return 0

    requests = []
    sent = 0
    new_pending = []

    for update in pending_fire_updates:
        retries_left = update.get("retries_left", FIRE_UPDATE_RELAY_CYCLES)
        if retries_left <= 0:
            continue
        location = update["location"]
        already_delivered = set(update.get("delivered_to", []))
        delivered = 0
        for peer in peer_states:
            if peer["rank"] == rank:
                continue
            if peer["rank"] == update.get("last_hop"):
                continue
            if peer["rank"] in already_delivered:
                continue
            if manhattan(location, peer["position"]) > radio_range:
                continue

            payload = {
                "type": "fire_update",
                "update_id": update["update_id"],
                "sender_rank": rank,
                "source_rank": update.get("source_rank", rank),
                "location": location,
                "step": update["step"],
                "last_hop": rank,
                "retries_left": retries_left - 1,
            }
            requests.append(comm.isend(payload, dest=peer["rank"], tag=TAG_FIRE_UPDATE))
            delivered += 1
            sent += 1
            already_delivered.add(peer["rank"])

            if events is not None:
                events.append(
                    {
                        "type": "communication",
                        "kind": "fire-update",
                        "from_rank": rank,
                        "to_rank": peer["rank"],
                        "from_role": "water",
                        "to_role": peer["role"],
                        "from": location,
                        "to": peer["position"],
                        "cell_count": 1,
                    }
                )

        # If isolated, keep retrying without decay until a relay path appears.
        if delivered == 0:
            new_pending.append(
                {
                    "update_id": update["update_id"],
                    "location": location,
                    "step": update["step"],
                    "source_rank": update.get("source_rank", rank),
                    "last_hop": update.get("last_hop", rank),
                    "retries_left": retries_left,
                    "delivered_to": list(already_delivered),
                }
            )
        # If we delivered this update at least once, drop it locally to prevent relay ping-pong.
        # Receivers continue propagation from their side when needed.

    if requests:
        MPI.Request.Waitall(requests)

    pending_fire_updates[:] = new_pending
    return sent


def nearest_position(source, positions):
    if not positions:
        return None
    return min(positions, key=lambda p: manhattan(source, p))


def autonomous_target_for_water(rank, drone, peer_states):
    # Use the true map only as a fallback when gossip has gone silent.
    pos = (drone.x, drone.y)
    fires = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 1]
    waters = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 3]
    chargers = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 2]
    water_peers = [p for p in peer_states if p["role"] == "water"]

    if drone.battery < 10 and chargers:
        return nearest_position(pos, chargers)
    if drone.water == 0 and waters:
        return nearest_position(pos, waters)
    if not fires or not water_peers:
        return None

    # Deterministically assign each fire to the nearest water drone.
    assigned = []
    for fire in fires:
        owner = min(water_peers, key=lambda p: (manhattan(p["position"], fire), p["rank"]))
        if owner["rank"] == rank:
            assigned.append(fire)

    return nearest_position(pos, assigned)


def coordinated_fire_target(rank, drone, peer_states, fires):
    water_peers = sorted([p for p in peer_states if p["role"] == "water"], key=lambda p: p["rank"])
    fires = sorted(fires)
    if not fires or not water_peers:
        return None

    assignments = {}
    fire_load = {fire: 0 for fire in fires}
    for peer in water_peers:
        target = min(
            fires,
            key=lambda fire: (
                fire_load[fire],
                manhattan(peer["position"], fire),
                fire,
            ),
        )
        assignments[peer["rank"]] = target
        fire_load[target] += 1

    return assignments.get(rank)


def coordinated_resource_target(rank, position, peer_states, targets):
    water_peers = [p for p in peer_states if p["role"] == "water"]
    if not targets or not water_peers:
        return None

    assigned = []
    for target in targets:
        owner = min(water_peers, key=lambda p: (manhattan(p["position"], target), p["rank"]))
        if owner["rank"] == rank:
            assigned.append(target)
    return nearest_position(position, assigned)


def compute_next_cell(drone, target):
    if target is None or not drone.can_use_battery():
        return None
    path = drone.route(target)
    if not path or len(path) <= 1:
        return None
    return path[1]


def alternate_water_candidate(drone, target, occupied_now):
    if target is None:
        return None

    rows, cols = len(drone.grid), len(drone.grid[0])
    current = (drone.x, drone.y)
    candidates = []

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        next_pos = (drone.x + dx, drone.y + dy)
        row, col = next_pos
        if not (0 <= row < rows and 0 <= col < cols):
            continue
        if next_pos in occupied_now:
            continue
        if drone.grid[row][col] != 0 and next_pos != target:
            continue

        dist = manhattan(next_pos, target)
        if dist <= manhattan(current, target):
            candidates.append((dist, next_pos))

    if not candidates:
        return None

    return min(candidates)[1]


def nearest_unknown(drone, grid_size):
    best = None
    best_dist = float("inf")
    start = (drone.x, drone.y)

    for row in range(grid_size):
        for col in range(grid_size):
            if (row, col) in drone.known_map:
                continue

            dist = manhattan(start, (row, col))
            if dist < best_dist:
                best = (row, col)
                best_dist = dist

    return best


def add_spiral_cell(path, seen, row, col, bounds):
    top, bottom, left, right = bounds
    if top <= row <= bottom and left <= col <= right and (row, col) not in seen:
        path.append((row, col))
        seen.add((row, col))


def add_axis_waypoints(path, seen, fixed, start, end, is_row, stride, bounds):
    step = 1 if end >= start else -1
    values = list(range(start, end + step, step * stride))
    if not values or values[-1] != end:
        values.append(end)

    for value in values:
        row, col = (fixed, value) if is_row else (value, fixed)
        add_spiral_cell(path, seen, row, col, bounds)


def build_ring_waypoints(bounds, stride):
    top, bottom, left, right = bounds
    path = []
    seen = set()
    stride = max(1, stride)

    add_axis_waypoints(path, seen, top, left, right, True, stride, bounds)
    if top < bottom:
        add_axis_waypoints(path, seen, right, top + 1, bottom, False, stride, bounds)
    if left < right and top < bottom:
        add_axis_waypoints(path, seen, bottom, right - 1, left, True, stride, bounds)
    if left < right and top + 1 < bottom:
        add_axis_waypoints(path, seen, left, bottom - 1, top + 1, False, stride, bounds)

    return path


def build_concentric_spiral_path(bounds, inward_step):
    rings = []
    top, bottom, left, right = bounds
    inward_step = max(1, inward_step)

    while top <= bottom and left <= right:
        ring_bounds = (top, bottom, left, right)
        ring = build_ring_waypoints(ring_bounds, inward_step)
        if ring:
            rings.append(ring)

        top += inward_step
        bottom -= inward_step
        left += inward_step
        right -= inward_step

    path = []
    for ring in rings:
        path.extend(ring)
    for ring in reversed(rings[:-1]):
        path.extend(reversed(ring))
    return path


def init_recon_spiral(drone, grid_size, sense_radius, recon_count, role_index):
    bounds = recon_partition_bounds(grid_size, recon_count, role_index)
    inward_step = max(1, sense_radius)
    drone.spiral_bounds = bounds
    drone.spiral_path = build_concentric_spiral_path(bounds, inward_step)
    drone.spiral_index = 0


def is_recon_patrol_cell_safe(drone, pos):
    row, col = pos
    return drone.grid[row][col] not in (1, 2)


def next_spiral_target(drone):
    path = getattr(drone, "spiral_path", [])
    checked = 0

    while path and checked < len(path):
        if getattr(drone, "spiral_index", 0) >= len(path):
            drone.spiral_index = 0

        target = path[drone.spiral_index]

        if (drone.x, drone.y) == target:
            drone.spiral_index += 1
            checked += 1
            continue

        if not is_recon_patrol_cell_safe(drone, target):
            drone.spiral_index += 1
            checked += 1
            continue

        route = drone.route(target)
        if route and len(route) > 1:
            return target

        drone.spiral_index += 1
        checked += 1

    return None


def move_one_step(drone, target):
    if target is None or not drone.can_use_battery():
        return False

    path = drone.route(target)
    if not path or len(path) <= 1:
        return False

    drone.x, drone.y = path[1]
    drone.battery_usage()
    return True


def recon_step(drone, grid_size, sense_radius, recon_count, role_index):
    old_pos = (drone.x, drone.y)
    refresh_known_cells(drone)
    sense_cells(drone, sense_radius)

    if not hasattr(drone, "spiral_path"):
        init_recon_spiral(drone, grid_size, sense_radius, recon_count, role_index)

    target = next_spiral_target(drone)
    if target is None:
        return False, old_pos, old_pos, None

    moved = move_one_step(drone, target)
    if (drone.x, drone.y) == target:
        drone.spiral_index += 1

    return moved, old_pos, (drone.x, drone.y), target


def water_step(
    comm,
    rank,
    drone,
    step,
    peer_states,
    radio_range,
    seen_fire_updates,
    pending_fire_updates,
    events,
    autonomous_mode=False,
):
    refresh_known_cells(drone)
    old_pos = (drone.x, drone.y)
    fires_before = fire_count(drone.grid)
    result = {
        "fire_update_sent": 0,
        "moved": False,
        "old_pos": old_pos,
        "new_pos": old_pos,
        "battery_spent": 0,
    }

    drone.extinguish()
    if fire_count(drone.grid) < fires_before:
        fire_update_sent, payload = broadcast_fire_update(
            comm=comm,
            rank=rank,
            location=old_pos,
            step=step,
            peer_states=peer_states,
            radio_range=radio_range,
            seen_fire_updates=seen_fire_updates,
            events=events,
        )
        events.append(
            {
                "type": "extinguish",
                "rank": rank,
                "role": "water",
                "drone_id": drone.id,
                "position": old_pos,
            }
        )
        print(
            f"[Step {step}] Rank {rank} WaterDrone {drone.id} extinguished fire at {old_pos} "
            f"and gossiped update to {fire_update_sent} nearby drones",
            flush=True,
        )
        pending_fire_updates.append(
            {
                "update_id": payload["update_id"],
                "location": payload["location"],
                "step": payload["step"],
                "source_rank": payload["source_rank"],
                "last_hop": rank,
                "retries_left": FIRE_UPDATE_RELAY_CYCLES,
                "delivered_to": [],
            }
        )
        result["fire_update_sent"] = fire_update_sent
        return result

    drone.refill_water()
    pos = (drone.x, drone.y)

    if autonomous_mode:
        fires = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 1]
        waters = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 3]
        chargers = [(r, c) for r, row in enumerate(drone.grid) for c, val in enumerate(row) if val == 2]
    else:
        fires = [p for p, v in drone.known_map.items() if v == 1]
        waters = [p for p, v in drone.known_map.items() if v == 3]
        chargers = [p for p, v in drone.known_map.items() if v == 2]

    if drone.battery < 10:
        target = coordinated_resource_target(rank, pos, peer_states, chargers) or nearest_position(pos, chargers)
    elif drone.water == 0:
        target = coordinated_resource_target(rank, pos, peer_states, waters) or nearest_position(pos, waters)
    else:
        target = coordinated_fire_target(rank, drone, peer_states, fires)
        if target is None and fires:
            # Fast reassignment: if this drone has no owned fire this cycle,
            # immediately assist at the nearest known fire instead of idling.
            target = nearest_position(pos, fires)
        if target is None and autonomous_mode:
            target = autonomous_target_for_water(rank, drone, peer_states)
            if target is not None:
                print(
                    f"[Step {step}] Rank {rank} WaterDrone {drone.id} switched to autonomous response "
                    f"and picked target {target}",
                    flush=True,
                )

    candidate = compute_next_cell(drone, target)
    occupied_now = {p["position"] for p in peer_states if p["rank"] != rank}
    moved = False
    if candidate is not None and candidate in occupied_now:
        blocked_candidate = candidate
        candidate = alternate_water_candidate(drone, target, occupied_now)
        if candidate is not None:
            print(
                f"[Step {step}] Rank {rank} WaterDrone {drone.id} detoured around "
                f"occupied cell {blocked_candidate} via {candidate}",
                flush=True,
            )
        else:
            print(
                f"[Step {step}] Rank {rank} WaterDrone {drone.id} held to avoid collision at {blocked_candidate}",
                flush=True,
            )

    if candidate is not None:
        drone.x, drone.y = candidate
        drone.battery_usage()
        moved = True
        result["moved"] = True
        result["new_pos"] = candidate
        result["battery_spent"] = 1

    if moved:
        events.append(
            {
                "type": "move",
                "rank": rank,
                "role": "water",
                "drone_id": drone.id,
                "from": old_pos,
                "to": (drone.x, drone.y),
                "target": target,
            }
        )
        print(
            f"[Step {step}] Rank {rank} WaterDrone {drone.id} moved "
            f"{old_pos} -> {(drone.x, drone.y)} target={target} "
            f"water={drone.water} battery={drone.battery}",
            flush=True,
        )
    else:
        events.append(
            {
                "type": "hold",
                "rank": rank,
                "role": "water",
                "drone_id": drone.id,
                "position": old_pos,
                "target": target,
            }
        )
        print(
            f"[Step {step}] Rank {rank} WaterDrone {drone.id} holding "
            f"at {old_pos} target={target}",
            flush=True,
        )

    return result


def role_for_rank(rank, recon_count):
    if rank < recon_count:
        return "recon", rank
    return "water", rank - recon_count


def setup_visualization(grid_size, sense_radius):

    import pygame
    pygame.init()
    pygame.font.init()

    info = pygame.display.Info()
    screen_w = int(info.current_w * 0.88)
    screen_h = int(info.current_h * 0.88)

    SIDEBAR = 230
    LOG_H = 110
    CELL = min(
        (screen_w - SIDEBAR) // grid_size,
        (screen_h - LOG_H)   // grid_size,
    )
    W = grid_size * CELL + SIDEBAR
    H = grid_size * CELL + LOG_H

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("MPI Drone Gossip Simulator")

    try:
        font_mono = pygame.font.SysFont("dejavusansmono", 12)
        font_bold = pygame.font.SysFont("dejavusansmono", 13, bold=True)
        font_title = pygame.font.SysFont("dejavusans", 14, bold=True)
        font_legend = pygame.font.SysFont("dejavusans", 12)
    except Exception:
        font_mono = pygame.font.SysFont(None, 14)
        font_bold = pygame.font.SysFont(None, 14)
        font_title = pygame.font.SysFont(None, 15)
        font_legend = pygame.font.SysFont(None, 13)

    CELL_COLORS = {
        0: (22, 101, 52),    # empty – deep green
        1: (185, 28, 28),    # fire  – red
        2: (234, 179, 8),    # charging – amber
        3: (29, 78, 216),    # water – blue
    }

    visual = {
        "pygame": pygame,
        "screen": screen,
        "font_mono": font_mono,
        "font_bold": font_bold,
        "font_title": font_title,
        "font_legend": font_legend,
        "grid_size": grid_size,
        "sense_radius": sense_radius,
        "CELL": CELL,
        "SIDEBAR": SIDEBAR,
        "LOG_H": LOG_H,
        "W": W,
        "H": H,
        "CELL_COLORS": CELL_COLORS,
        "backend": "pygame",
        "event_log": [],
    }
    return visual


def flatten_event_groups(event_groups):
    events = []
    for group in event_groups:
        events.extend(group)
    return events


def build_fire_targets_map(events, states):
    """Build mapping of discovered fires to water drones targeting them.
    Returns dict: {(row, col): [list of ranks targeting this fire]}
    Uses fires discovered by recon drones (from their known_map) instead of actual grid.
    """
    # Extract all discovered fires from recon drones' known_maps
    discovered_fires = set()
    for state in states:
        if state['role'] == 'recon':
            known_map = state.get('known_map', {})
            for pos, val in known_map.items():
                if val == 1:  # 1 = fire
                    discovered_fires.add(pos)
    
    # Initialize fire_targets with all discovered fires (even if no target)
    fire_targets = {fire: [] for fire in discovered_fires}
    
    # Track which water drones are targeting which discovered fires from current events
    for event in events:
        # Track water drone moves/holds with targets
        if event['type'] in ('move', 'hold') and event['role'] == 'water':
            target = event.get('target')
            if target is not None and target in fire_targets:
                # Use rank to identify the drone
                if event['rank'] not in fire_targets[target]:
                    fire_targets[target].append(event['rank'])
    
    return fire_targets


def plot_point(position):
    return position[1], position[0]


def draw_arrow(ax, start, end, color, linestyle="-", alpha=0.75, linewidth=2.0, rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    start_xy = plot_point(start)
    end_xy = plot_point(end)
    if start_xy == end_xy:
        return

    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="->",
        mutation_scale=14,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=5,
    )
    ax.add_patch(arrow)


def event_text(event):
    if event["type"] == "move":
        label = "R" if event["role"] == "recon" else "W"
        return f"{label}{event['rank']} move {event['from']} -> {event['to']}"
    if event["type"] == "hold":
        label = "R" if event["role"] == "recon" else "W"
        return f"{label}{event['rank']} hold at {event['position']}"
    if event["type"] == "extinguish":
        return f"W{event['rank']} extinguish {event['position']}"
    if event["type"] == "communication":
        return (
            f"{event['kind']} r{event['from_rank']} -> r{event['to_rank']} "
            f"({event['cell_count']} cells)"
        )
    return str(event)


def render_visualization(visual, grid, states, events, step, remaining, delay):
    import math
    pygame = visual["pygame"]
    screen = visual["screen"]
    # update screen ref in case user resized
    visual["screen"] = pygame.display.get_surface()
    screen = visual["screen"]
    CELL = visual["CELL"]
    SIDEBAR = visual["SIDEBAR"]
    LOG_H = visual["LOG_H"]
    W = visual["W"]
    H = visual["H"]
    grid_size = visual["grid_size"]
    CELL_COLORS = visual["CELL_COLORS"]
    font_mono = visual["font_mono"]
    font_bold = visual["font_bold"]
    font_title = visual["font_title"]
    font_legend = visual["font_legend"]

    BG         = (10,  13,  22)
    GRID_LINE  = (35,  40,  58)
    SIDEBAR_BG = (16,  19,  32)
    LOG_BG     = (10,  12,  20)
    WHITE      = (255, 255, 255)
    GRAY       = (155, 165, 185)
    SENSE_TINT = (100, 200, 255, 38)

    COM_COLORS = {
        "recon-to-water": (30,  120, 220),
        "water-to-water": (245, 130,   0),
        "fire-update":    (220,  50,  50),
    }

    GRID_W = grid_size * CELL
    GRID_H = grid_size * CELL

    # animation tick (cycles 0-59 each frame for flicker effects)
    tick = visual.get("_tick", 0)
    visual["_tick"] = (tick + 1) % 60

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
            return
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # only register clicks inside the grid area
            if mx < GRID_W and my < GRID_H:
                clicked_col = mx // CELL
                clicked_row = my // CELL
                if 0 <= clicked_row < grid_size and 0 <= clicked_col < grid_size:
                    if ev.button == 1:  # left click → place fire
                        if grid[clicked_row][clicked_col] == 0:
                            grid[clicked_row][clicked_col] = 1
                            visual.setdefault("user_placed_fires", []).append((clicked_row, clicked_col))
                            print(f"[Step {step}] User placed fire at ({clicked_row}, {clicked_col})", flush=True)
                    elif ev.button == 3:  # right click → remove fire
                        if grid[clicked_row][clicked_col] == 1:
                            grid[clicked_row][clicked_col] = 0
                            print(f"[Step {step}] User removed fire at ({clicked_row}, {clicked_col})", flush=True)

    screen.fill(BG)

    # ── terrain sprites ───────────────────────────────────────────────────────
    def draw_terrain(surface, r, c, val, cx, cy, tick):
        half = CELL // 2
        # ground texture – subtle grid squares already give structure

        if val == 1:  # 🔥 FIRE – hand-drawn flickering flames
            flame_colors = [
                (255, 60,  10),
                (255, 140,  0),
                (255, 220,  0),
                (255,  80, 20),
            ]
            # base ember glow
            pygame.draw.ellipse(surface, (180, 30, 0),
                                (cx - half + 6, cy + half - 10, CELL - 12, 10))
            # draw 3 flame tongues with tick-based wobble
            for fi, (ox, scale) in enumerate([(-7, 1.0), (0, 1.3), (7, 0.9)]):
                phase = (tick * 6 + fi * 20) % 360
                wobble = int(math.sin(math.radians(phase)) * 3)
                tip_x = cx + ox + wobble
                tip_y = cy - int(half * scale * 0.85)
                mid_y = cy + int(half * 0.1)
                color = flame_colors[fi % len(flame_colors)]
                pts = [
                    (tip_x, tip_y),
                    (cx + ox - 6, mid_y),
                    (cx + ox + 6, mid_y),
                ]
                pygame.draw.polygon(surface, color, pts)
                # inner bright core
                inner = flame_colors[(fi + 2) % len(flame_colors)]
                inner_pts = [
                    (tip_x, tip_y + 6),
                    (cx + ox - 3, mid_y),
                    (cx + ox + 3, mid_y),
                ]
                pygame.draw.polygon(surface, inner, inner_pts)

        elif val == 2:  # 🌳 TREE / OBSTACLE
                    # trunk
                    trunk_w = max(4, CELL // 7)
                    trunk_h = CELL // 3
                    pygame.draw.rect(surface, (90, 55, 20),
                                    (cx - trunk_w // 2, cy + CELL // 6,
                                    trunk_w, trunk_h), border_radius=2)
                    # three layered canopy circles (dark to light, bottom to top)
                    for layer, (oy, r, color) in enumerate([
                        ( CELL // 8,      CELL // 3,      (20,  90, 20)),
                        ( 0,              CELL // 3 - 2,  (30, 120, 30)),
                        (-CELL // 7,      CELL // 4,      (50, 160, 50)),
                        (-CELL // 4,      CELL // 6,      (80, 200, 80)),
                    ]):
                        pygame.draw.circle(surface, color, (cx, cy + oy), r)
                    # subtle highlight on top canopy
                    pygame.draw.circle(surface, (110, 220, 100),
                                    (cx - CELL // 10, cy - CELL // 4 - 2),
                                    CELL // 10)

        elif val == 3:  # 💧 WATER SOURCE
            # ripple rings
            for ring in range(3):
                r_phase = (tick * 4 + ring * 20) % 60
                r_radius = 6 + ring * 7 + r_phase // 10
                alpha_val = max(0, 180 - ring * 55 - r_phase * 2)
                if r_radius < half - 2 and alpha_val > 0:
                    pygame.draw.circle(surface, (80, 160, 255),
                                       (cx, cy), r_radius, 2)
            # solid water drop body
            drop_pts = [
                (cx, cy - half + 8),
                (cx - 8, cy + 4),
                (cx,     cy + half - 6),
                (cx + 8, cy + 4),
            ]
            pygame.draw.polygon(surface, (30, 100, 220), drop_pts)
            pygame.draw.polygon(surface, (120, 200, 255), drop_pts, 1)
            # shine dot
            pygame.draw.circle(surface, (200, 230, 255), (cx - 2, cy - 4), 2)

    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            base_color = CELL_COLORS.get(val, (40, 44, 58))
            # darken empty cells slightly for contrast
            if val == 0:
                base_color = (18, 52, 30)
            rect = pygame.Rect(c * CELL, r * CELL, CELL - 1, CELL - 1)
            pygame.draw.rect(screen, base_color, rect, border_radius=5)
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2
            draw_terrain(screen, r, c, val, cx, cy, tick)

    # ── sensing overlay ────────────────────────────────────────────────────────
    overlay = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
    for state in states:
        if state["role"] != "recon":
            continue
        cx2, cy2 = state["position"]
        for dx in range(-visual["sense_radius"], visual["sense_radius"] + 1):
            for dy in range(-visual["sense_radius"], visual["sense_radius"] + 1):
                nx, ny = cx2 + dx, cy2 + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    pygame.draw.rect(overlay, SENSE_TINT,
                                     (ny * CELL, nx * CELL, CELL - 1, CELL - 1),
                                     border_radius=4)
    screen.blit(overlay, (0, 0))

    # ── grid lines ─────────────────────────────────────────────────────────────
    for i in range(grid_size + 1):
        pygame.draw.line(screen, GRID_LINE, (i * CELL, 0), (i * CELL, GRID_H))
        pygame.draw.line(screen, GRID_LINE, (0, i * CELL), (GRID_W, i * CELL))

    # ── arrow helpers ──────────────────────────────────────────────────────────
    def cell_center(pos):
        r2, c2 = pos
        return (c2 * CELL + CELL // 2, r2 * CELL + CELL // 2)

    def draw_arrow(src_pos, dst_pos, color, dashed=False, offset=0):
        sx, sy = cell_center(src_pos)
        ex, ey = cell_center(dst_pos)
        if (sx, sy) == (ex, ey):
            return
        dx2, dy2 = ex - sx, ey - sy
        length = max((dx2**2 + dy2**2) ** 0.5, 1)
        nx2, ny2 = -dy2 / length * offset, dx2 / length * offset
        sx, sy = int(sx + nx2), int(sy + ny2)
        ex, ey = int(ex + nx2), int(ey + ny2)
        if dashed:
            steps = 10
            for i in range(0, steps, 2):
                x1 = int(sx + (ex - sx) * i / steps)
                y1 = int(sy + (ey - sy) * i / steps)
                x2 = int(sx + (ex - sx) * (i + 1) / steps)
                y2 = int(sy + (ey - sy) * (i + 1) / steps)
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)
        else:
            pygame.draw.line(screen, color, (sx, sy), (ex, ey), 2)
        angle = math.atan2(ey - sy, ex - sx)
        for side in (+0.4, -0.4):
            ax2 = int(ex - 11 * math.cos(angle + side))
            ay2 = int(ey - 11 * math.sin(angle + side))
            pygame.draw.line(screen, color, (ex, ey), (ax2, ay2), 2)

    comm_idx = 0
    for event in events:
        if event["type"] == "move" and event["from"] != event["to"]:
            draw_arrow(event["from"], event["to"], (60, 210, 90))
        elif event["type"] == "communication":
            kind = event["kind"]
            color = COM_COLORS.get(kind, (100, 120, 140))
            draw_arrow(event["from"], event["to"], color,
                       dashed=(kind == "fire-update"),
                       offset=6 * (1 if comm_idx % 2 == 0 else -1))
            comm_idx += 1

    # ── drone sprites ──────────────────────────────────────────────────────────
    def draw_recon_drone(surface, cx, cy, label):
        """Fixed-wing recon plane: fuselage + swept wings + tail."""
        body_color   = (170,  90, 230)
        wing_color   = (130,  55, 190)
        cockpit_col  = (180, 230, 255)
        exhaust_col  = (255, 160,  40)

        # glow halo
        glow = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        pygame.draw.circle(glow, (170, 90, 230, 55), (CELL//2, CELL//2), CELL//2 - 2)
        surface.blit(glow, (cx - CELL//2, cy - CELL//2))

        # fuselage (thin horizontal body)
        pygame.draw.ellipse(surface, body_color,
                            (cx - 14, cy - 4, 28, 8))

        # swept main wings
        left_wing  = [(cx - 2, cy - 2), (cx - 16, cy + 9), (cx - 8, cy + 9)]
        right_wing = [(cx + 2, cy - 2), (cx + 16, cy + 9), (cx + 8, cy + 9)]
        pygame.draw.polygon(surface, wing_color, left_wing)
        pygame.draw.polygon(surface, wing_color, right_wing)
        pygame.draw.polygon(surface, body_color, left_wing, 1)
        pygame.draw.polygon(surface, body_color, right_wing, 1)

        # tail fin
        tail_fin = [(cx - 14, cy - 2), (cx - 18, cy - 9), (cx - 10, cy - 2)]
        pygame.draw.polygon(surface, wing_color, tail_fin)

        # cockpit bubble
        pygame.draw.ellipse(surface, cockpit_col,
                            (cx - 4, cy - 5, 10, 7))

        # exhaust glow
        exhaust_r = 3 + int(math.sin(math.radians(tick * 9)) * 1.5)
        pygame.draw.circle(surface, exhaust_col, (cx - 15, cy + 1), exhaust_r)

        # label
        lsurf = font_bold.render(label, True, WHITE)
        surface.blit(lsurf, (cx - lsurf.get_width() // 2, cy - CELL // 2 + 1))

    def draw_water_drone(surface, cx, cy, label):
        """Quadcopter: central hub + 4 arms + rotors."""
        hub_color   = (40,  40,  50)
        arm_color   = (80,  90, 110)
        rotor_color = (100, 180, 255)
        body_color  = (50,  60,  80)

        rotor_spin = int((tick * 8) % 360)
        rotor_r = CELL // 2 - 6

        # 4 arms
        arm_len = CELL // 2 - 4
        for angle_deg in (45, 135, 225, 315):
            rad = math.radians(angle_deg)
            ex2 = int(cx + arm_len * math.cos(rad))
            ey2 = int(cy + arm_len * math.sin(rad))
            pygame.draw.line(surface, arm_color, (cx, cy), (ex2, ey2), 3)

            # rotor disk at arm tip
            pygame.draw.circle(surface, (30, 40, 60), (ex2, ey2), rotor_r)
            pygame.draw.circle(surface, rotor_color, (ex2, ey2), rotor_r, 2)

            # spinning rotor blade lines
            for blade in (0, 90):
                br = math.radians(rotor_spin + blade)
                bx1 = int(ex2 + (rotor_r - 1) * math.cos(br))
                by1 = int(ey2 + (rotor_r - 1) * math.sin(br))
                bx2 = int(ex2 - (rotor_r - 1) * math.cos(br))
                by2 = int(ey2 - (rotor_r - 1) * math.sin(br))
                pygame.draw.line(surface, (160, 220, 255), (bx1, by1), (bx2, by2), 2)

        # central body hex
        hub_pts = []
        for a in range(6):
            rad = math.radians(a * 60 + 30)
            hub_pts.append((int(cx + 8 * math.cos(rad)), int(cy + 8 * math.sin(rad))))
        pygame.draw.polygon(surface, body_color, hub_pts)
        pygame.draw.polygon(surface, arm_color,  hub_pts, 1)

        # water payload indicator (blue dot)
        pygame.draw.circle(surface, (60, 140, 255), (cx, cy), 4)

        # label
        lsurf = font_bold.render(label, True, WHITE)
        surface.blit(lsurf, (cx - lsurf.get_width() // 2, cy - CELL // 2 + 1))

    for state in states:
        cx, cy = cell_center(state["position"])
        if state["role"] == "recon":
            draw_recon_drone(screen, cx, cy, f"R{state['rank']}")
        else:
            draw_water_drone(screen, cx, cy, f"W{state['rank']}")

    # ── sidebar ────────────────────────────────────────────────────────────────
    SX = GRID_W + 4
    pygame.draw.rect(screen, SIDEBAR_BG, (SX, 0, SIDEBAR, H))

    def sb_text(txt, y, color=WHITE, font=None):
        f = font or font_legend
        s = f.render(txt, True, color)
        screen.blit(s, (SX + 10, y))
        return y + s.get_height() + 4

    y = 10
    y = sb_text(f"Step {step}", y, (255, 215, 70), font_title)
    fire_col = (255, 75, 75) if remaining > 0 else (70, 240, 110)
    y = sb_text(f"Fires left: {remaining}", y, fire_col, font_bold)
    y = sb_text(f"Sense r={visual['sense_radius']}", y, GRAY)
    y += 8

    y = sb_text("LEGEND", y, (100, 115, 150), font_bold)
    LEGEND = [
        ((170,  90, 230), "Recon Drone"),
        ((100, 180, 255), "Water quadcopter"),
        ((100, 200, 255), "Sense radius"),
        (( 60, 210,  90), "Movement"),
        (COM_COLORS["recon-to-water"], "Recon to water"),
        (COM_COLORS["water-to-water"], "Water to water"),
        (COM_COLORS["fire-update"],    "Fire update"),
    ]
    CELL_MAP = [
        ((18,  52, 30),  "Empty"),
        ((185, 28, 28),  "Fire"),
        ((234,179,  8),  "Obstacle/Tree"),
        (( 29, 78,216),  "Water"),
    ]
    for color, lbl in LEGEND:
        pygame.draw.rect(screen, color, (SX + 10, y + 3, 13, 13), border_radius=3)
        s = font_legend.render(lbl, True, GRAY)
        screen.blit(s, (SX + 28, y))
        y += s.get_height() + 4
    y += 4
    y = sb_text("GRID", y, (100, 115, 150), font_bold)
    for color, lbl in CELL_MAP:
        pygame.draw.rect(screen, color, (SX + 10, y + 3, 13, 13), border_radius=3)
        s = font_legend.render(lbl, True, GRAY)
        screen.blit(s, (SX + 28, y))
        y += s.get_height() + 4
    y += 6

    # ── identified fires and targeting drones ──
    fire_targets = build_fire_targets_map(events, states)
    if fire_targets:
        y = sb_text("FIRES", y, (100, 115, 150), font_bold)
        for fire_pos in sorted(fire_targets.keys()):
            targeting_ranks = fire_targets[fire_pos]
            if targeting_ranks:
                water_labels = ", ".join(f"W{r}" for r in sorted(targeting_ranks))
                detail = f"{fire_pos} -> {water_labels}"
                target_color = (255, 150, 150)  # lighter red when targeted
            else:
                detail = f"{fire_pos} -> No target"
                target_color = (220, 100, 100)  # darker red when not targeted
            y = sb_text(detail, y, target_color, font_mono)
            if y > H - LOG_H - 20:
                break
        y += 4

    y = sb_text("DRONES", y, (100, 115, 150), font_bold)
    for state in sorted(states, key=lambda s: s["rank"]):
        lbl = "R" if state["role"] == "recon" else "W"
        bat = state.get("battery", "?")
        wat = state.get("water", "")
        pos = state["position"]
        bat_str = f"{bat:,}" if isinstance(bat, int) else str(bat)
        detail = f"{lbl}{state['rank']} {pos} b={bat_str}"
        if wat not in (None, ""):
            detail += f" w={wat}"
        col = (170, 90, 230) if state["role"] == "recon" else (100, 180, 255)
        y = sb_text(detail, y, col, font_mono)
        if y > H - LOG_H - 20:
            break

    # ── event log ─────────────────────────────────────────────────────────────
    LY = GRID_H + 2
    pygame.draw.rect(screen, LOG_BG, (0, LY, GRID_W, LOG_H))
    pygame.draw.line(screen, GRID_LINE, (0, LY), (GRID_W, LY), 1)
    title_s = font_bold.render(f"Step {step} Events", True, (255, 215, 70))
    screen.blit(title_s, (8, LY + 5))

    lines = [event_text(ev) for ev in events] or ["No events this step"]
    visual["event_log"] = lines
    MAX_LINES = 4
    COL_W = GRID_W // 2
    for i, txt in enumerate(lines[:MAX_LINES]):
        s = font_mono.render(txt[:50], True, GRAY)
        screen.blit(s, (8, LY + 22 + i * 18))
    for i, txt in enumerate(lines[MAX_LINES: MAX_LINES * 2]):
        s = font_mono.render(txt[:50], True, GRAY)
        screen.blit(s, (COL_W + 8, LY + 22 + i * 18))
    if len(lines) > MAX_LINES * 2:
        more = font_mono.render(f"... +{len(lines) - MAX_LINES * 2} more", True, (90, 100, 120))
        screen.blit(more, (8, LY + 22 + MAX_LINES * 18))

    pygame.display.flip()
    pygame.time.wait(int(delay * 1000))

def finish_visualization(visual, keep_open):
    pygame = visual["pygame"]
    if keep_open:
        print("Close the pygame window to finish the MPI program.", flush=True)
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
    else:
        pygame.time.wait(1000)
    pygame.quit()


def open_mini_log(path):
    log_path = os.path.abspath(path)
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    handle.write("MPI Gossip Compact Debug Log\n")
    handle.write("=" * 32 + "\n")
    return handle, log_path


def compact_positions_text(states):
    ordered = sorted(states, key=lambda s: s["rank"])
    parts = []
    for state in ordered:
        label = "R" if state["role"] == "recon" else "W"
        pos = state["position"]
        parts.append(f"{label}{state['rank']}@({pos[0]},{pos[1]})")
    return " ".join(parts)


def key_events_text(events, max_items):
    keys = []
    for event in events:
        if event["type"] == "extinguish":
            keys.append(f"W{event['rank']} extinguish {event['position']}")
        elif event["type"] == "communication" and event.get("kind") == "fire-update":
            keys.append(f"fire-update r{event['from_rank']}->r{event['to_rank']}")
    if not keys:
        return "none"
    return " | ".join(keys[:max_items])


def write_mini_log_step(
    handle,
    step,
    remaining,
    global_comm_events,
    no_comm_streak,
    synced_changes,
    states,
    events,
    max_event_items,
):
    handle.write(
        f"[Step {step}] fires={remaining} comm={global_comm_events} "
        f"silent={no_comm_streak} sync={synced_changes}\n"
    )
    handle.write(f"  pos: {compact_positions_text(states)}\n")
    handle.write(f"  key: {key_events_text(events, max_event_items)}\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser(description="MPI gossip communication demo")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--fires", type=int, default=8)
    parser.add_argument("--waters", type=int, default=4)
    parser.add_argument("--charging", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recon-drones", type=int, default=1)
    parser.add_argument("--sense-radius", type=int, default=3, help="Grid/fire sensing radius for recon drones")
    parser.add_argument("--radio-range", type=int, default=3, help="MPI gossip range between drones")
    parser.add_argument("--delay", type=float, default=0.45, help="Matplotlib frame delay")
    parser.add_argument("--no-ui", action="store_true", help="Run MPI simulation without the matplotlib graph")
    parser.add_argument("--keep-open", action="store_true", help="Keep the matplotlib window open after the last step")
    parser.add_argument(
        "--mini-log",
        action="store_true",
        help=f"Write a compact debug log on rank 0 (default file: {DEFAULT_MINI_LOG})",
    )
    parser.add_argument(
        "--mini-log-file",
        type=str,
        default="",
        help="Path for compact debug log file (rank 0 only)",
    )
    parser.add_argument(
        "--mini-log-events",
        type=int,
        default=8,
        help="Max key events per step in compact debug log",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if args.recon_drones < 1:
        raise ValueError("Use at least one recon drone")
    if world_size <= args.recon_drones:
        if rank == 0:
            print("Run with more MPI processes than recon drones, e.g. mpiexec -n 4 python mpi_gossip_sim.py")
        return

    grid = build_world(args.grid_size, args.fires, args.waters, args.charging, args.seed)
    tracked_fire_cells = initial_fire_cells(grid)
    start_positions = pick_start_positions(grid, world_size, args.seed, args.recon_drones)
    role, role_index = role_for_rank(rank, args.recon_drones)
    drone = make_drone(rank, role, role_index, start_positions[rank], grid)
    seen_messages = set()
    seen_fire_updates = set()
    pending_fire_updates = []
    mini_log_requested = args.mini_log or bool(args.mini_log_file)
    mini_log_path = args.mini_log_file if args.mini_log_file else DEFAULT_MINI_LOG
    mini_log_handle = None
    visual = None
    if rank == 0 and not args.no_ui:
        try:
            visual = setup_visualization(args.grid_size, args.sense_radius)
        except Exception as exc:
            print(f"UI disabled: could not initialize matplotlib window ({exc})", flush=True)
    if rank == 0 and mini_log_requested:
        try:
            mini_log_handle, resolved_path = open_mini_log(mini_log_path)
            print(f"Compact debug log: {resolved_path}", flush=True)
        except OSError as exc:
            print(f"Mini log disabled: could not open file ({exc})", flush=True)
            mini_log_handle = None

    print(
        f"Rank {rank} started as {role} drone_id={drone.id} "
        f"at {(drone.x, drone.y)}",
        flush=True,
    )
    if rank == 0:
        print(
            f"Recon sense radius: {args.sense_radius}; "
            f"water drones do not sense; MPI communication radius: {args.radio_range}",
            flush=True,
        )
        if visual is not None:
            print(
                f"UI backend in use: {visual['backend']}",
                flush=True,
            )

    mission_success = False
    no_comm_streak = 0

    for step in range(1, args.steps + 1):
        local_events = []
        local_comm_events = 0
        local_move_info = {
            "moved": False,
            "old_pos": (drone.x, drone.y),
            "battery_spent": 0,
            "role": role,
            "target": None,
        }
        peer_states = comm.allgather(
            {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
            }
        )

        received = drain_messages(
            comm,
            drone,
            grid,
            seen_messages,
            seen_fire_updates,
            pending_fire_updates,
            rank,
            step,
        )
        local_comm_events += received["gossip"] + received["fire_update"]
        comm.Barrier()

        if role == "recon":
            moved, old_pos, new_pos, target = recon_step(
                drone,
                args.grid_size,
                args.sense_radius,
                args.recon_drones,
                role_index,
            )
            if moved:
                local_move_info["moved"] = True
                local_move_info["old_pos"] = old_pos
                local_move_info["battery_spent"] = 10
                local_move_info["target"] = target
            local_events.append(
                {
                    "type": "move" if moved else "hold",
                    "rank": rank,
                    "role": role,
                    "drone_id": drone.id,
                    "from": old_pos,
                    "to": new_pos,
                    "position": new_pos,
                    "target": target,
                }
            )
            sent = send_gossip(
                comm=comm,
                rank=rank,
                role=role,
                drone=drone,
                peer_states=peer_states,
                target_roles={"water"},
                radio_range=args.radio_range,
                step=step,
                reason="recon-to-water",
                events=local_events,
            )
            print(
                f"[Step {step}] Rank {rank} RecDrone {drone.id} "
                f"{'moved' if moved else 'held'} at {(drone.x, drone.y)} "
                f"and sent gossip to {sent} water drones",
                flush=True,
            )
            local_comm_events += sent

        comm.Barrier()
        received = drain_messages(
            comm,
            drone,
            grid,
            seen_messages,
            seen_fire_updates,
            pending_fire_updates,
            rank,
            step,
        )
        local_comm_events += received["gossip"] + received["fire_update"]
        comm.Barrier()

        peer_states_after_recon = comm.allgather(
            {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
            }
        )

        forwarded = forward_fire_updates(
            comm=comm,
            rank=rank,
            peer_states=peer_states_after_recon,
            radio_range=args.radio_range,
            pending_fire_updates=pending_fire_updates,
            events=local_events,
        )
        local_comm_events += forwarded

        comm.Barrier()

        if role == "water":
            water_result = water_step(
                comm=comm,
                rank=rank,
                drone=drone,
                step=step,
                peer_states=peer_states_after_recon,
                radio_range=args.radio_range,
                seen_fire_updates=seen_fire_updates,
                pending_fire_updates=pending_fire_updates,
                events=local_events,
                autonomous_mode=False,
            )
            local_comm_events += water_result["fire_update_sent"]
            local_move_info["moved"] = water_result["moved"]
            local_move_info["old_pos"] = water_result["old_pos"]
            local_move_info["battery_spent"] = water_result["battery_spent"]

        comm.Barrier()
        peer_states_after_water = comm.allgather(
            {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
            }
        )

        # Resolve same-cell collisions deterministically: lowest rank keeps the cell.
        cell_to_ranks = {}
        for p in peer_states_after_water:
            cell_to_ranks.setdefault(p["position"], []).append(p["rank"])
        loser_ranks = set()
        for ranks in cell_to_ranks.values():
            if len(ranks) > 1:
                for loser in sorted(ranks)[1:]:
                    loser_ranks.add(loser)

        if rank in loser_ranks and local_move_info["moved"]:
            current_pos = (drone.x, drone.y)
            drone.x, drone.y = local_move_info["old_pos"]
            drone.battery = min(drone.batterymax, drone.battery + local_move_info["battery_spent"])
            local_move_info["moved"] = False
            print(
                f"[Step {step}] Rank {rank} {role.capitalize()}Drone {drone.id} rollback "
                f"{current_pos} -> {(drone.x, drone.y)} due to collision",
                flush=True,
            )
            local_events.append(
                {
                    "type": "hold",
                    "rank": rank,
                    "role": role,
                    "drone_id": drone.id,
                    "position": (drone.x, drone.y),
                }
            )

        comm.Barrier()
        peer_states_after_collision = comm.allgather(
            {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
            }
        )

        if role == "water":
            sent_to_water = send_gossip(
                comm=comm,
                rank=rank,
                role=role,
                drone=drone,
                peer_states=peer_states_after_collision,
                target_roles={"water"},
                radio_range=args.radio_range,
                step=step,
                reason="water-to-water",
                events=local_events,
            )
            print(
                f"[Step {step}] Rank {rank} WaterDrone {drone.id} sent gossip "
                f"to {sent_to_water} water drones",
                flush=True,
            )
            local_comm_events += sent_to_water

        comm.Barrier()
        received = drain_messages(
            comm,
            drone,
            grid,
            seen_messages,
            seen_fire_updates,
            pending_fire_updates,
            rank,
            step,
        )
        local_comm_events += received["gossip"] + received["fire_update"]
        forwarded = forward_fire_updates(
            comm=comm,
            rank=rank,
            peer_states=peer_states_after_collision,
            radio_range=args.radio_range,
            pending_fire_updates=pending_fire_updates,
            events=local_events,
        )
        local_comm_events += forwarded

        # Reliable status sync: each water drone reports all tracked fire cells.
        local_fire_snapshot = (
            water_fire_status_snapshot(grid, tracked_fire_cells)
            if role == "water"
            else None
        )
        water_snapshots = comm.allgather(local_fire_snapshot)
        synced_changes = reconcile_fire_status_from_waters(grid, tracked_fire_cells, water_snapshots)
        if rank == 0 and synced_changes > 0:
            print(
                f"[Step {step}] Recon 0 synchronized {synced_changes} fire-cell updates from water drones",
                flush=True,
            )

        global_comm_events = comm.allreduce(local_comm_events, op=MPI.SUM)
        if global_comm_events == 0:
            no_comm_streak += 1
        else:
            no_comm_streak = 0

        remaining = comm.allreduce(fire_count(grid), op=MPI.MAX)

        collect_for_root = (not args.no_ui) or mini_log_requested
        all_states = None
        flattened_events = None
        if collect_for_root:
            local_state = {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
                "battery": drone.battery,
                "water": getattr(drone, "water", None),
            }
            # Include recon drone's known_map for visualization
            if role == "recon":
                local_state["known_map"] = drone.known_map
            all_states = comm.gather(local_state, root=0)
            all_event_groups = comm.gather(local_events, root=0)
            if rank == 0:
                flattened_events = flatten_event_groups(all_event_groups)
                if not args.no_ui and visual is not None:
                    render_visualization(
                        visual=visual,
                        grid=grid,
                        states=all_states,
                        events=flattened_events,
                        step=step,
                        remaining=remaining,
                        delay=args.delay,
                    )

        # ── inject user-placed fires into all drones ──────────────────────────
        if rank == 0 and visual is not None:
            new_fires = visual.pop("user_placed_fires", [])
        else:
            new_fires = []
        new_fires = comm.bcast(new_fires, root=0)
        for fr, fc in new_fires:
            grid[fr][fc] = 1
            drone.grid[fr][fc] = 1
            
            if (fr, fc) not in tracked_fire_cells:
                tracked_fire_cells.append((fr, fc))

        if rank == 0:
            print(f"[Step {step}] MPI mission remaining fires: {remaining}", flush=True)
            print(
                f"[Step {step}] Communication events this step: {global_comm_events} "
                f"(silent streak: {no_comm_streak})",
                flush=True,
            )
            if mini_log_handle is not None and all_states is not None and flattened_events is not None:
                write_mini_log_step(
                    handle=mini_log_handle,
                    step=step,
                    remaining=remaining,
                    global_comm_events=global_comm_events,
                    no_comm_streak=no_comm_streak,
                    synced_changes=synced_changes,
                    states=all_states,
                    events=flattened_events,
                    max_event_items=max(1, args.mini_log_events),
                )
            if no_comm_streak == FALLBACK_AFTER_SILENT_CYCLES:
                print(
                    f"[Step {step}] No communication for {FALLBACK_AFTER_SILENT_CYCLES} cycles. "
                    "Water drones will use autonomous nearest-target response from next step.",
                    flush=True,
                )

        if remaining == 0:
            mission_success = True
            if rank == 0:
                print(f"MPI mission success in {step} steps", flush=True)
            break

        comm.Barrier()

    if rank == 0 and not mission_success:
        print(f"MPI mission ended after reaching the --steps limit ({args.steps}).", flush=True)

    if rank == 0 and mini_log_handle is not None:
        mini_log_handle.write(
            f"Mission {'SUCCESS' if mission_success else 'INCOMPLETE'} after max steps={args.steps}\n"
        )
        mini_log_handle.flush()
        mini_log_handle.close()

    if rank == 0 and visual is not None:
        finish_visualization(visual, keep_open=args.keep_open)


if __name__ == "__main__":
    main()
