
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
