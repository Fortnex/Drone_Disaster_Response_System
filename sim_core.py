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
FIRE_UPDATE_RELAY_CYCLES = 6


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
