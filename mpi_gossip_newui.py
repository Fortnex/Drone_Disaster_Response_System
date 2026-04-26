import argparse

from mpi4py import MPI

from sim_core import (
    build_world,
    drain_messages,
    fire_count,
    forward_fire_updates,
    initial_fire_cells,
    make_drone,
    pick_start_positions,
    recon_step,
    reconcile_fire_status_from_waters,
    role_for_rank,
    send_gossip,
    water_fire_status_snapshot,
    water_step,
)
from ui import finish_visualization, flatten_event_groups, render_visualization, setup_visualization


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
    parser.add_argument("--delay", type=float, default=0.45, help="Pygame frame delay")
    parser.add_argument("--no-ui", action="store_true", help="Run MPI simulation without the Pygame UI")
    parser.add_argument("--keep-open", action="store_true", help="Keep the Pygame window open after the last step")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()



    grid = build_world(args.grid_size, args.fires, args.waters, args.charging, args.seed)
    tracked_fire_cells = initial_fire_cells(grid)
    start_positions = pick_start_positions(grid, world_size, args.seed, args.recon_drones)
    role, role_index = role_for_rank(rank, args.recon_drones)
    drone = make_drone(rank, role, role_index, start_positions[rank], grid)
    seen_messages = set()
    seen_fire_updates = set()
    pending_fire_updates = []
    visual = None
    if rank == 0 and not args.no_ui:
        try:
            visual = setup_visualization(args.grid_size, args.sense_radius)
        except Exception as exc:
            print(f"UI disabled: could not initialize Pygame window ({exc})", flush=True)

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

        collect_for_root = not args.no_ui
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
