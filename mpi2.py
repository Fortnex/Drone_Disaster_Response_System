# mpi_gossip_sim.py (patched: spiral recon + delta gossip + local fire updates)

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


COMMUNICATION_COLORS = {
    "recon-to-water": "#1565C0",
    "water-to-water": "#F57C00",
    "fire-update": "#D32F2F",
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def fire_count(grid):
    return sum(1 for row in grid for cell in row if cell == 1)


def build_world(grid_size, fires, waters, charging, seed):
    random.seed(seed)
    world = MakeGrid(fire=fires, water=waters, gridSize=grid_size, charging=charging)
    return world.generate()


def pick_start_positions(grid, count, seed):
    rng = random.Random(seed + 1000)
    empty_cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == 0
    ]
    rng.shuffle(empty_cells)

    if len(empty_cells) < count:
        raise ValueError("Not enough empty cells to place all drones")

    return empty_cells[:count]


def make_drone(rank, role, role_index, start, grid):
    if role == "recon":
        d = RecDrone(
            id=90 + role_index,
            battery=250,
            batterymax=250,
            x=start[0],
            y=start[1],
            grid=grid,
        )
        # initialize timestamped map
        d.known_map = {}
        d.last_sent = {}
        d.spiral_path = None
        d.spiral_index = 0
        return d

    drone = WaterDrone(
        id=10 + role_index,
        battery=220,
        batterymax=220,
        water=0,
        watermax=4,
        x=start[0],
        y=start[1],
        grid=grid,
    )
    # timestamped map
    drone.known_map = {(drone.x, drone.y): (grid[drone.x][drone.y], 0)}
    drone.last_sent = {}
    return drone


# -------- DELTA GOSSIP (timestamped) --------

def known_cells_payload(known_map, last_sent, peer_id):
    payload = []
    last_time = last_sent.get(peer_id, -1)

    for (r, c), (val, ts) in known_map.items():
        if ts > last_time:
            payload.append((r, c, val, ts))

    return payload


def merge_cells(known_map, cells):
    changed = 0
    for row, col, value, ts in cells:
        pos = (row, col)
        if pos not in known_map or ts > known_map[pos][1]:
            known_map[pos] = (value, ts)
            changed += 1
    return changed


def refresh_known_cells(drone, step):
    for pos in list(drone.known_map.keys()):
        val = drone.grid[pos[0]][pos[1]]
        old = drone.known_map.get(pos)
        if old is None or val != old[0]:
            drone.known_map[pos] = (val, step)


def sense_cells(drone, radius, step):
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx = drone.x + dx
            ny = drone.y + dy
            if 0 <= nx < len(drone.grid) and 0 <= ny < len(drone.grid[0]):
                drone.known_map[(nx, ny)] = (drone.grid[nx][ny], step)


# -------- MESSAGING --------

def drain_messages(comm, drone, grid, seen_messages, rank, step):
    status = MPI.Status()

    while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_GOSSIP, status=status):
        source = status.Get_source()
        message = comm.recv(source=source, tag=TAG_GOSSIP)
        mid = message["message_id"]
        if mid in seen_messages:
            continue
        seen_messages.add(mid)

        changed = merge_cells(drone.known_map, message["cells"])

        print(f"[Step {step}] Rank {rank} received gossip from {source}: {changed} updates", flush=True)

    while comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_FIRE_UPDATE, status=status):
        source = status.Get_source()
        message = comm.recv(source=source, tag=TAG_FIRE_UPDATE)
        row, col = message["location"]
        grid[row][col] = 0
        drone.known_map[(row, col)] = (0, step)


# -------- LOCAL GOSSIP --------

def send_gossip(comm, rank, drone, peer_states, radio_range, step):
    sent = 0
    sender_pos = (drone.x, drone.y)

    for peer in peer_states:
        if peer["rank"] == rank:
            continue
        if manhattan(sender_pos, peer["position"]) > radio_range:
            continue

        cells = known_cells_payload(drone.known_map, drone.last_sent, peer["rank"])
        if not cells:
            continue

        payload = {
            "message_id": f"{rank}:{step}:{peer['rank']}",
            "cells": cells,
        }

        comm.isend(payload, dest=peer["rank"], tag=TAG_GOSSIP)
        drone.last_sent[peer["rank"]] = step
        sent += 1

    return sent


# -------- LOCAL FIRE UPDATE --------

def broadcast_fire_update(comm, rank, location, step, peer_states, radio_range):
    for peer in peer_states:
        if peer["rank"] == rank:
            continue
        if manhattan(location, peer["position"]) > radio_range:
            continue

        payload = {
            "location": location,
            "step": step,
        }

        comm.isend(payload, dest=peer["rank"], tag=TAG_FIRE_UPDATE)


# -------- SPIRAL RECON --------

def init_spiral(drone, grid_size, step):
    path = []
    top, bottom, left, right = 0, grid_size - 1, 0, grid_size - 1

    while top <= bottom and left <= right:
        for y in range(left, right + 1, step):
            path.append((top, y))
        for x in range(top, bottom + 1, step):
            path.append((x, right))
        for y in range(right, left - 1, -step):
            path.append((bottom, y))
        for x in range(bottom, top - 1, -step):
            path.append((x, left))

        top += step
        bottom -= step
        left += step
        right -= step

    drone.spiral_path = path
    drone.spiral_index = 0


def move_one_step(drone, target):
    if target is None:
        return False
    path = drone.route(target)
    if not path or len(path) <= 1:
        return False
    drone.x, drone.y = path[1]
    drone.battery_usage()
    return True


def recon_step(drone, grid_size, sense_radius, step):
    old_pos = (drone.x, drone.y)

    refresh_known_cells(drone, step)
    sense_cells(drone, sense_radius, step)

    if drone.spiral_path is None:
        init_spiral(drone, grid_size, 2 * sense_radius)

    if drone.spiral_index < len(drone.spiral_path):
        target = drone.spiral_path[drone.spiral_index]
        if (drone.x, drone.y) == target:
            drone.spiral_index += 1
        else:
            move_one_step(drone, target)
    else:
        target = None

    return old_pos, (drone.x, drone.y), target


# -------- WATER --------

def water_step(comm, rank, drone, step, peer_states, radio_range):
    refresh_known_cells(drone, step)

    # extinguish
    if drone.grid[drone.x][drone.y] == 1:
        drone.grid[drone.x][drone.y] = 0
        drone.known_map[(drone.x, drone.y)] = (0, step)
        broadcast_fire_update(comm, rank, (drone.x, drone.y), step, peer_states, radio_range)
        return

    # pick nearest fire
    fires = [pos for pos, (val, _) in drone.known_map.items() if val == 1]
    if not fires:
        return

    target = min(fires, key=lambda f: manhattan((drone.x, drone.y), f))
    move_one_step(drone, target)


# -------- MAIN --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--fires", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sense-radius", type=int, default=3)
    parser.add_argument("--radio-range", type=int, default=3)
    parser.add_argument("--recon-drones", type=int, default=1)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    grid = build_world(args.grid_size, args.fires, 4, 3, args.seed)
    start_positions = pick_start_positions(grid, world_size, args.seed)

    role = "recon" if rank < args.recon_drones else "water"
    drone = make_drone(rank, role, rank, start_positions[rank], grid)

    seen_messages = set()

    for step in range(1, args.steps + 1):
        peer_states = comm.allgather({
            "rank": rank,
            "position": (drone.x, drone.y),
        })

        drain_messages(comm, drone, grid, seen_messages, rank, step)

        if role == "recon":
            recon_step(drone, args.grid_size, args.sense_radius, step)

        if role == "water":
            water_step(comm, rank, drone, step, peer_states, args.radio_range)

        send_gossip(comm, rank, drone, peer_states, args.radio_range, step)

        remaining = comm.allreduce(fire_count(grid), op=MPI.MAX)

        if rank == 0:
            print(f"[Step {step}] Fires remaining: {remaining}")

        if remaining == 0:
            break


if __name__ == "__main__":
    main()
