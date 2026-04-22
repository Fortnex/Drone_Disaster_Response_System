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


def drain_messages(comm, drone, grid, seen_messages, rank, step):
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
        row, col = message["location"]
        grid[row][col] = 0
        drone.known_map[(row, col)] = 0
        received["fire_update"] += 1

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


def broadcast_fire_update(comm, rank, location, step, world_size, peer_states=None, events=None):
    payload = {
        "type": "fire_update",
        "sender_rank": rank,
        "location": location,
        "step": step,
    }
    requests = [
        comm.isend(payload, dest=dest, tag=TAG_FIRE_UPDATE)
        for dest in range(world_size)
        if dest != rank
    ]

    if requests:
        MPI.Request.Waitall(requests)

    if events is not None and peer_states is not None:
        for peer in peer_states:
            if peer["rank"] == rank:
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


def water_step(comm, rank, world_size, drone, step, peer_states, events):
    refresh_known_cells(drone)
    old_pos = (drone.x, drone.y)
    fires_before = fire_count(drone.grid)

    drone.extinguish()
    if fire_count(drone.grid) < fires_before:
        broadcast_fire_update(
            comm=comm,
            rank=rank,
            location=old_pos,
            step=step,
            world_size=world_size,
            peer_states=peer_states,
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
            f"[Step {step}] Rank {rank} WaterDrone {drone.id} extinguished fire at {old_pos}",
            flush=True,
        )
        return

    drone.refill_water()
    target = drone.decide_target()
    moved = move_one_step(drone, target)

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


def role_for_rank(rank, recon_count):
    if rank < recon_count:
        return "recon", rank
    return "water", rank - recon_count


def setup_visualization(grid_size):
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
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
        mlines.Line2D([], [], color="#1B5E20", linewidth=2, label="Movement"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["recon-to-water"], linewidth=2, label="Recon to water"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["water-to-water"], linewidth=2, label="Water to water"),
        mlines.Line2D([], [], color=COMMUNICATION_COLORS["fire-update"], linewidth=2, linestyle="--", label="Fire update"),
    ]

    return {
        "plt": plt,
        "fig": fig,
        "ax": ax,
        "log_ax": log_ax,
        "cmap": cmap,
        "norm": norm,
        "legend_items": legend_items,
        "grid_size": grid_size,
        "backend": plt.get_backend().lower(),
    }


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
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="#111111", linestyle="-", linewidth=0.65)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_title(f"MPI Drone Gossip | Step {step} | Fires left: {remaining}")

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
    if "agg" in visual["backend"]:
        visual["fig"].canvas.draw()
    else:
        visual["plt"].draw()
        visual["plt"].pause(delay)


def finish_visualization(visual, keep_open):
    visual["plt"].ioff()
    if "agg" in visual["backend"]:
        visual["plt"].close(visual["fig"])
    elif keep_open:
        print("Close the matplotlib window to finish the MPI program.", flush=True)
        visual["plt"].show()
    else:
        visual["plt"].pause(1.0)
        visual["plt"].close(visual["fig"])


def main():
    parser = argparse.ArgumentParser(description="MPI gossip communication demo")
    parser.add_argument("--steps", type=int, default=40)
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if args.recon_drones < 1:
        raise ValueError("Use at least one recon drone")
    if world_size <= args.recon_drones:
        if rank == 0:
            print("Run with more MPI processes than recon drones, e.g. mpiexec -n 4 python3 mpi_gossip_sim.py")
        return

    grid = build_world(args.grid_size, args.fires, args.waters, args.charging, args.seed)
    start_positions = pick_start_positions(grid, world_size, args.seed)
    role, role_index = role_for_rank(rank, args.recon_drones)
    drone = make_drone(rank, role, role_index, start_positions[rank], grid)
    seen_messages = set()
    visual = setup_visualization(args.grid_size) if rank == 0 and not args.no_ui else None

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

    mission_success = False

    for step in range(1, args.steps + 1):
        local_events = []
        peer_states = comm.allgather(
            {
                "rank": rank,
                "role": role,
                "drone_id": drone.id,
                "position": (drone.x, drone.y),
            }
        )

        drain_messages(comm, drone, grid, seen_messages, rank, step)
        comm.Barrier()

        if role == "recon":
            moved, old_pos, new_pos, target = recon_step(drone, args.grid_size, args.sense_radius)
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

        comm.Barrier()
        drain_messages(comm, drone, grid, seen_messages, rank, step)
        comm.Barrier()

        if role == "water":
            water_step(
                comm=comm,
                rank=rank,
                world_size=world_size,
                drone=drone,
                step=step,
                peer_states=peer_states,
                events=local_events,
            )

        comm.Barrier()
        peer_states_after_water = comm.allgather(
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
                peer_states=peer_states_after_water,
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

        comm.Barrier()
        drain_messages(comm, drone, grid, seen_messages, rank, step)
        remaining = comm.allreduce(fire_count(grid), op=MPI.MAX)

        if not args.no_ui:
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
                render_visualization(
                    visual=visual,
                    grid=grid,
                    states=all_states,
                    events=flatten_event_groups(all_event_groups),
                    step=step,
                    remaining=remaining,
                    delay=args.delay,
                )

        if rank == 0:
            print(f"[Step {step}] MPI mission remaining fires: {remaining}", flush=True)

        if remaining == 0:
            mission_success = True
            if rank == 0:
                print(f"MPI mission success in {step} steps", flush=True)
            break

        comm.Barrier()

    if rank == 0 and not mission_success:
        print(f"MPI mission ended after reaching the --steps limit ({args.steps}).", flush=True)

    if rank == 0 and visual is not None:
        finish_visualization(visual, keep_open=args.keep_open)


if __name__ == "__main__":
    main()
