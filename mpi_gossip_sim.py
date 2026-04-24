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
        return RecDrone(
            id=90 + role_index,
            battery=250,
            batterymax=250,
            x=start[0],
            y=start[1],
            grid=grid,
        )

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
    water_peers = [p for p in peer_states if p["role"] == "water"]
    if not fires or not water_peers:
        return None

    assigned = []
    for fire in fires:
        owner = min(water_peers, key=lambda p: (manhattan(p["position"], fire), p["rank"]))
        if owner["rank"] == rank:
            assigned.append(fire)
    return nearest_position((drone.x, drone.y), assigned)


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


def move_one_step(drone, target):
    if target is None or not drone.can_use_battery():
        return False

    path = drone.route(target)
    if not path or len(path) <= 1:
        return False

    drone.x, drone.y = path[1]
    drone.battery_usage()
    return True


def recon_step(drone, grid_size, sense_radius):
    old_pos = (drone.x, drone.y)
    refresh_known_cells(drone)
    sense_cells(drone, sense_radius)
    target = nearest_unknown(drone, grid_size)
    moved = move_one_step(drone, target)
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
    if candidate is not None and candidate not in occupied_now:
        drone.x, drone.y = candidate
        drone.battery_usage()
        moved = True
        result["moved"] = True
        result["new_pos"] = candidate
        result["battery_spent"] = 1
    elif candidate is not None and candidate in occupied_now:
        print(
            f"[Step {step}] Rank {rank} WaterDrone {drone.id} held to avoid collision at {candidate}",
            flush=True,
        )

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
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(figsize=(13, 7))
    layout = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.45])
    ax = fig.add_subplot(layout[0])
    log_ax = fig.add_subplot(layout[1])

    cmap = mcolors.ListedColormap(["#2E7D32", "#D32F2F", "#FFD54F", "#1976D2"])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    legend_items = [
        mlines.Line2D([], [], color="#7B1FA2", marker="^", linestyle="None", markersize=10, label="Recon drone"),
        mlines.Line2D([], [], color="#212121", marker="o", linestyle="None", markersize=9, label="Water drone"),
        mpatches.Patch(facecolor="#81D4FA", alpha=0.30, label="Recon sensing radius"),
        mlines.Line2D([], [], color="#1B5E20", linewidth=2, label="Movement"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["recon-to-water"], linewidth=2, label="Recon to water"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["water-to-water"], linewidth=2, label="Water to water"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["fire-update"], linewidth=2, linestyle="--", label="Fire update"),
    ]

    visual = {
        "plt": plt,
        "fig": fig,
        "ax": ax,
        "log_ax": log_ax,
        "cmap": cmap,
        "norm": norm,
        "radius_cmap": mcolors.ListedColormap(["#81D4FA"]),
        "legend_items": legend_items,
        "grid_size": grid_size,
        "sense_radius": sense_radius,
        "backend": plt.get_backend().lower(),
    }
    return visual


def flatten_event_groups(event_groups):
    events = []
    for group in event_groups:
        events.extend(group)
    return events


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
    import numpy as np

    ax = visual["ax"]
    log_ax = visual["log_ax"]
    grid_size = visual["grid_size"]

    ax.clear()
    log_ax.clear()

    ax.imshow(np.array(grid), cmap=visual["cmap"], norm=visual["norm"])

    # Shade recon sensing radius to match sensing logic.
    radius_mask = np.zeros((grid_size, grid_size))
    for state in states:
        if state["role"] != "recon":
            continue
        cx, cy = state["position"]
        for dx in range(-visual["sense_radius"], visual["sense_radius"] + 1):
            for dy in range(-visual["sense_radius"], visual["sense_radius"] + 1):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    radius_mask[nx][ny] = 1
    masked_radius = np.ma.masked_where(radius_mask == 0, radius_mask)
    ax.imshow(masked_radius, cmap=visual["radius_cmap"], alpha=0.30, vmin=1, vmax=1, zorder=1.5)

    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="#111111", linestyle="-", linewidth=0.65)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_title(
        f"MPI Drone Gossip | Step {step} | Fires left: {remaining} | "
        f"Recon sense radius: {visual['sense_radius']}"
    )

    movement_index = 0
    communication_index = 0
    for event in events:
        if event["type"] == "move":
            draw_arrow(
                ax,
                event["from"],
                event["to"],
                color="#1B5E20",
                linewidth=2.4,
                alpha=0.95,
                rad=0.0,
            )
            movement_index += 1
        elif event["type"] == "communication":
            kind = event["kind"]
            draw_arrow(
                ax,
                event["from"],
                event["to"],
                color=COMMUNICATION_COLORS.get(kind, "#455A64"),
                linestyle="--" if kind == "fire-update" else "-",
                linewidth=1.8,
                alpha=0.68,
                rad=0.08 if communication_index % 2 == 0 else -0.08,
            )
            communication_index += 1

    for state in states:
        x, y = plot_point(state["position"])
        if state["role"] == "recon":
            color = "#7B1FA2"
            marker = "^"
            label = f"R{state['rank']}"
        else:
            color = "#212121"
            marker = "o"
            label = f"W{state['rank']}"

        ax.scatter(x, y, s=160, c=color, marker=marker, edgecolors="white", linewidths=1.3, zorder=8)
        ax.text(x + 0.12, y - 0.12, label, color="white", fontsize=8, weight="bold", zorder=9)

    ax.legend(handles=visual["legend_items"], loc="upper right", fontsize=8, framealpha=0.92)

    log_ax.axis("off")
    log_ax.set_title("Step Events", loc="left")
    lines = [event_text(event) for event in events]
    if not lines:
        lines = ["No movement or communication this step"]
    text = "\n".join(lines[:34])
    if len(lines) > 34:
        text += f"\n... {len(lines) - 34} more events"
    log_ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=8)

    visual["fig"].tight_layout()
    visual["plt"].draw()
    visual["plt"].pause(delay)


def finish_visualization(visual, keep_open):
    visual["plt"].ioff()
    if keep_open:
        print("Close the matplotlib window to finish the MPI program.", flush=True)
        visual["plt"].show()
    else:
        visual["plt"].pause(1.0)
        visual["plt"].close(visual["fig"])


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
    start_positions = pick_start_positions(grid, world_size, args.seed)
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
            moved, old_pos, new_pos, target = recon_step(drone, args.grid_size, args.sense_radius)
            if moved:
                local_move_info["moved"] = True
                local_move_info["old_pos"] = old_pos
                local_move_info["battery_spent"] = 1
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
                autonomous_mode=no_comm_streak >= FALLBACK_AFTER_SILENT_CYCLES,
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
