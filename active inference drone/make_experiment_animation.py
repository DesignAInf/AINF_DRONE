"""Animate the actual flight paths in the two V5 drone experiments.

The simulator has one physical navigation coordinate: cross-track displacement
``y``. The horizontal axis in this animation is therefore normalized mission
progress, not a fabricated second spatial state. Part I draws the recorded
single-drone path. Part II draws both recorded vehicle paths and their payload
mean. Wind, control, causal gap, design events, agreement, and outcomes are all
read from representative successful frozen missions.

Author: Luca M. Possati
"""

from __future__ import annotations

import argparse
from math import sin
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from active_drone.collective import make_collective_system, run_collective_mission
from active_drone.simulation import make_system, run_mission


W, H = 1200, 675
BG = "#F3F6FA"
WHITE = "#FFFFFF"
INK = "#102A43"
MUTED = "#64778B"
FAINT = "#DFE8F0"
GRID = "#EAF0F5"
BLUE = "#1769E0"
BLUE_BG = "#EDF5FF"
PURPLE = "#7656E8"
PURPLE_BG = "#F3EFFF"
GREEN = "#11875D"
GREEN_BG = "#EAF8F2"
ORANGE = "#F28E2B"
ORANGE_BG = "#FFF5E9"
RED = "#D64545"
RED_BG = "#FFF1F0"
CYAN = "#00A7A5"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


F10, F11, F12, F14 = (font(s) for s in (10, 11, 12, 14))
B10, B11, B12, B14, B16, B21, B27 = (font(s, True) for s in (10, 11, 12, 14, 16, 21, 27))


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
             fill: str, fnt: ImageFont.FreeTypeFont) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              text, fill=fill, font=fnt)


def rounded(draw: ImageDraw.ImageDraw, box, radius=16, fill=WHITE,
            outline=FAINT, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(draw: ImageDraw.ImageDraw, box, text: str, color: str,
         fill: str, fnt=B11) -> None:
    rounded(draw, box, radius=12, fill=fill, outline=color, width=1)
    centered(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), text, color, fnt)


def blend(a: str, b: str, t: float) -> str:
    t = float(np.clip(t, 0, 1))
    av = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    bv = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round((1-t)*x+t*y):02x}" for x, y in zip(av, bv))


def drone(draw: ImageDraw.ImageDraw, x: float, y: float, color: str,
          scale: float = 1.0, label: str | None = None) -> None:
    arm, rotor = 28 * scale, 9 * scale
    line_w = max(2, int(4 * scale))
    draw.line((x - arm, y - 10 * scale, x + arm, y + 10 * scale), fill=INK, width=line_w)
    draw.line((x - arm, y + 10 * scale, x + arm, y - 10 * scale), fill=INK, width=line_w)
    for rx, ry in ((x - arm, y - 10 * scale), (x + arm, y + 10 * scale),
                   (x - arm, y + 10 * scale), (x + arm, y - 10 * scale)):
        draw.ellipse((rx - rotor, ry - 2, rx + rotor, ry + 2), fill=MUTED)
    draw.rounded_rectangle((x - 17 * scale, y - 10 * scale, x + 17 * scale, y + 10 * scale),
                           radius=max(3, int(6 * scale)), fill=color, outline=INK,
                           width=max(1, int(2 * scale)))
    draw.ellipse((x - 4 * scale, y - 4 * scale, x + 4 * scale, y + 4 * scale), fill=WHITE)
    if label:
        centered(draw, (x, y + 24 * scale), label, INK, B10)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float],
          delta: tuple[float, float], color: str, width: int = 3) -> None:
    x0, y0 = start
    x1, y1 = x0 + delta[0], y0 + delta[1]
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    length = max((delta[0] ** 2 + delta[1] ** 2) ** .5, 1e-8)
    ux, uy = delta[0] / length, delta[1] / length
    px, py = -uy, ux
    size = 7
    points = [(x1, y1),
              (x1 - size * ux + .55 * size * px, y1 - size * uy + .55 * size * py),
              (x1 - size * ux - .55 * size * px, y1 - size * uy - .55 * size * py)]
    draw.polygon(points, fill=color)


def path_xy(values: np.ndarray, index: int, box: tuple[int, int, int, int],
            y_limit: float = 1.15) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    scale = (y1 - y0) * .44 / y_limit
    n = max(len(values) - 1, 1)
    # Leave enough room for the drone and its blanket halo at start and goal.
    start, end = x0 + 48, x1 - 48
    return [(start + (end - start) * j / n, cy - scale * float(values[j]))
            for j in range(min(index + 1, len(values)))]


def draw_track(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], accent: str) -> None:
    x0, y0, x1, y1 = box
    cy = (y0 + y1) / 2
    rounded(draw, box, radius=16, fill="#FBFDFE", outline=FAINT, width=1)
    # The visible band is the operational corridor; collision limits lie beyond it.
    draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y0 + 24), fill=RED_BG)
    draw.rectangle((x0 + 1, y1 - 24, x1 - 1, y1 - 1), fill=RED_BG)
    for fraction in (.25, .50, .75):
        x = x0 + fraction * (x1 - x0)
        draw.line((x, y0 + 10, x, y1 - 10), fill=GRID, width=1)
    draw.line((x0 + 8, cy, x1 - 8, cy), fill="#BFCEDA", width=2)
    for x in np.linspace(x0 + 16, x1 - 16, 18):
        draw.line((x, cy, x + 7, cy), fill=WHITE, width=2)
    draw.text((x0 + 10, y0 + 7), "+ cross-track", fill="#A15A5A", font=F10)
    draw.text((x0 + 10, y1 - 19), "− cross-track", fill="#A15A5A", font=F10)
    draw.text((x0 + 10, cy + 7), "reference route", fill=MUTED, font=F10)
    # Start and goal are mission-progress markers, not additional simulated states.
    draw.ellipse((x0 + 8, cy - 7, x0 + 22, cy + 7), fill=WHITE, outline=accent, width=2)
    draw.ellipse((x1 - 24, cy - 10, x1 - 4, cy + 10), fill=WHITE, outline=accent, width=2)
    draw.ellipse((x1 - 19, cy - 5, x1 - 9, cy + 5), fill=accent)
    draw.text((x0 + 7, y1 + 5), "START", fill=MUTED, font=B10)
    draw.text((x1 - 34, y1 + 5), "GOAL", fill=MUTED, font=B10)


def trail(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str,
          width: int = 4) -> None:
    if len(points) < 2:
        return
    # A pale under-stroke separates the trajectory from grid and event markers.
    draw.line(points, fill=WHITE, width=width + 4, joint="curve")
    draw.line(points, fill=color, width=width, joint="curve")
    for p in points[::max(1, len(points) // 14)]:
        draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color)


def event_indices(records: list[dict]) -> list[int]:
    return [j for j, r in enumerate(records) if bool(r["design_changed_causal_matrix"])]


def event_markers(draw: ImageDraw.ImageDraw, records: list[dict], values: np.ndarray,
                  idx: int, box: tuple[int, int, int, int], color: str,
                  collective: bool = False) -> None:
    for number, j in enumerate(event_indices(records), start=1):
        if j > idx:
            continue
        point = path_xy(values, j, box)[-1]
        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10),
                     fill=WHITE, outline=color, width=3)
        centered(draw, point, f"{'J' if collective else 'S'}{number}", color, B10)


def boundary_halo(draw: ImageDraw.ImageDraw, center: tuple[float, float],
                  gap: float, initial_gap: float, radius: tuple[float, float],
                  phase: float, collective: bool = False) -> None:
    strength = float(np.clip(1 - gap / max(initial_gap, 1e-9), 0, 1))
    color = blend("#9BAABA", GREEN, strength)
    pulse = 2 + int(2 * ((1 + sin(phase)) / 2))
    x, y = center
    rx, ry = radius
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=color, width=pulse)
    if collective:
        draw.arc((x - rx - 5, y - ry - 5, x + rx + 5, y + ry + 5),
                 190, 350, fill=BLUE, width=2)
        draw.arc((x - rx - 5, y - ry - 5, x + rx + 5, y + ry + 5),
                 10, 170, fill=PURPLE, width=2)


def pipeline(draw: ImageDraw.ImageDraw, labels: tuple[str, ...], active: int,
             x0: int, x1: int, y: int, accent: str) -> None:
    centers = np.linspace(x0, x1, len(labels))
    draw.line((centers[0], y, centers[-1], y), fill=FAINT, width=4)
    if active > 0:
        draw.line((centers[0], y, centers[active], y), fill=accent, width=4)
    for j, (x, label) in enumerate(zip(centers, labels)):
        current = j == active
        radius = 8 if current else 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=accent if j <= active else WHITE,
                     outline=accent if j <= active else "#C8D4E0", width=2)
        centered(draw, (x, y + 21), label, accent if current else MUTED,
                 B10 if current else F10)


def count_changes(records: list[dict], idx: int) -> int:
    return sum(bool(r["design_changed_causal_matrix"]) for r in records[:idx + 1])


def individual_stage(idx: int, records: list[dict]) -> int:
    if idx < 7:
        return 0
    if idx < 11:
        return 1
    if count_changes(records, idx) < 3:
        return 2
    return 3


def collective_stage(idx: int, records: list[dict]) -> int:
    if idx < 15:
        return 0
    if idx < 24:
        return 1
    if count_changes(records, idx) < 2:
        return 2
    return 3


def panel_header(draw: ImageDraw.ImageDraw, x: int, part: str, title: str,
                 subtitle: str, accent: str) -> None:
    draw.text((x + 20, 128), part, fill=accent, font=B11)
    draw.text((x + 20, 147), title, fill=INK, font=B21)
    draw.text((x + 20, 174), subtitle, fill=MUTED, font=F11)


def individual_panel(draw: ImageDraw.ImageDraw, records: list[dict], idx: int,
                     frame_no: int) -> None:
    px0, px1 = 24, 588
    rounded(draw, (px0, 116, px1, 625), radius=22)
    panel_header(draw, px0, "EXPERIMENT I", "The drone redesigns while flying",
                 "Recorded cross-track path under wind and active control.", BLUE)
    stage = individual_stage(idx, records)
    pipeline(draw, ("OBSERVE", "INFER", "SELECT", "ENACT"), stage,
             px0 + 58, px1 - 58, 209, BLUE)

    box = (45, 250, 567, 467)
    draw_track(draw, box, BLUE)
    ys = np.asarray([r["y"] for r in records], float)
    points = path_xy(ys, idx, box)
    trail(draw, points, BLUE, 4)
    event_markers(draw, records, ys, idx, box, ORANGE)
    current = records[idx]
    gaps = np.asarray([r["factorization_gap"] for r in records], float)
    x, y = points[-1]
    boundary_halo(draw, (x, y), gaps[idx], gaps[0], (38, 27), frame_no * .55)
    wind = float(current["wind"])
    control = float(current["control"])
    arrow(draw, (x - 46, y), (0, -30 * wind), RED, 3)
    arrow(draw, (x - 34, y), (0, -25 * control), GREEN, 3)
    drone(draw, x, y, BLUE, .62, "D")

    if bool(current["design_changed_causal_matrix"]):
        status, color, fill = f"EFE ENACTS {str(current['design_action']).upper()}", ORANGE, ORANGE_BG
    elif stage == 3:
        status, color, fill = "BOUNDARY CONSTRUCTED — FLIGHT REMAINS VIABLE", GREEN, GREEN_BG
    elif stage == 2:
        status, color, fill = "EFE COMPARES MOTOR + STRUCTURAL POLICIES", BLUE, BLUE_BG
    elif stage == 1:
        status, color, fill = "INFERRING THE BLANKET FROM THE FLIGHT HISTORY", PURPLE, PURPLE_BG
    else:
        status, color, fill = "ACCUMULATING PATH EVIDENCE", MUTED, BG
    pill(draw, (93, 490, 539, 520), status, color, fill, B10)

    draw.text((46, 541), "LATERAL POSITION", fill=MUTED, font=B10)
    draw.text((46, 558), f"{ys[idx]:+.2f}", fill=INK, font=B16)
    draw.text((168, 541), "WIND", fill=MUTED, font=B10)
    draw.text((168, 558), f"{wind:+.2f}", fill=RED, font=B16)
    draw.text((257, 541), "CONTROL", fill=MUTED, font=B10)
    draw.text((257, 558), f"{control:+.2f}", fill=GREEN, font=B16)
    draw.text((359, 541), "CAUSAL GAP", fill=MUTED, font=B10)
    draw.text((359, 558), f"{gaps[idx]:.2f}", fill=GREEN if gaps[idx] < .05 else RED, font=B16)
    draw.text((465, 541), "SHIELDS", fill=MUTED, font=B10)
    draw.text((465, 558), f"{count_changes(records, idx)} / 3", fill=INK, font=B16)
    draw.text((46, 594), "red arrow: wind   green arrow: selected motor control", fill=MUTED, font=F10)
    draw.text((507, 594), "seed 1000", fill="#91A0AF", font=F10)


def collective_panel(draw: ImageDraw.ImageDraw, records: list[dict], idx: int,
                     frame_no: int) -> None:
    px0, px1 = 612, 1176
    rounded(draw, (px0, 116, px1, 625), radius=22)
    panel_header(draw, px0, "EXPERIMENT II", "The pair flies as one coordinated system",
                 "Recorded paths of A, B, and their shared payload.", PURPLE)
    stage = collective_stage(idx, records)
    pipeline(draw, ("LOCAL", "POOL", "AGREE", "ENACT"), stage,
             px0 + 58, px1 - 58, 209, PURPLE)

    box = (633, 250, 1155, 467)
    draw_track(draw, box, PURPLE)
    pair = np.asarray([r["y"] for r in records], float)
    payload = np.asarray([r["payload"] for r in records], float)
    path_a = path_xy(pair[:, 0], idx, box)
    path_b = path_xy(pair[:, 1], idx, box)
    path_p = path_xy(payload, idx, box)
    trail(draw, path_a, BLUE, 3)
    trail(draw, path_b, PURPLE, 3)
    trail(draw, path_p, ORANGE, 5)
    event_markers(draw, records, payload, idx, box, ORANGE, collective=True)

    current = records[idx]
    gaps = np.asarray([r["factorization_gap"] for r in records], float)
    xa, ya = path_a[-1]
    xb, yb = path_b[-1]
    xp, yp = path_p[-1]
    separation = max(abs(ya - yb), 18)
    boundary_halo(draw, (xp, yp), gaps[idx], gaps[0], (43, separation / 2 + 25),
                  frame_no * .55, collective=True)
    draw.line((xa, ya, xb, yb), fill=ORANGE, width=4)
    draw.rounded_rectangle((xp - 10, yp - 7, xp + 10, yp + 7), radius=3,
                           fill=ORANGE, outline=INK, width=1)
    drone(draw, xa, ya, BLUE, .48, "A")
    drone(draw, xb, yb, PURPLE, .48, "B")
    wind = np.asarray(current["wind"], float)
    arrow(draw, (xp - 48, yp), (0, -24 * float(wind.mean())), RED, 3)

    if bool(current["design_changed_causal_matrix"]):
        status, color, fill = "MATCHING POLICIES CHANGE THE SHARED CAUSAL STRUCTURE", ORANGE, ORANGE_BG
    elif stage == 3:
        status, color, fill = "GROUP BOUNDARY ACTIVE — JOINT FLIGHT REMAINS VIABLE", GREEN, GREEN_BG
    elif bool(current["agreement"]):
        status, color, fill = f"HANDSHAKE: {str(current['proposal_a']).upper()}", BLUE, BLUE_BG
    elif stage == 1:
        status, color, fill = "COMPLEMENTARY FLIGHT EVIDENCE IS POOLED", PURPLE, PURPLE_BG
    else:
        status, color, fill = "LOCAL MODELS SEE DIFFERENT PARTS OF THE PATH", MUTED, BG
    pill(draw, (681, 490, 1127, 520), status, color, fill, B10)

    formation = float(current["formation_error"])
    draw.text((634, 541), "PAYLOAD", fill=MUTED, font=B10)
    draw.text((634, 558), f"{payload[idx]:+.2f}", fill=ORANGE, font=B16)
    draw.text((744, 541), "FORMATION ERROR", fill=MUTED, font=B10)
    draw.text((744, 558), f"{formation:+.2f}", fill=INK, font=B16)
    draw.text((901, 541), "GROUP GAP", fill=MUTED, font=B10)
    draw.text((901, 558), f"{gaps[idx]:.2f}", fill=GREEN if gaps[idx] < .05 else RED, font=B16)
    draw.text((1012, 541), "JOINT SHIELDS", fill=MUTED, font=B10)
    draw.text((1012, 558), f"{count_changes(records, idx)} / 2", fill=INK, font=B16)
    draw.ellipse((635, 591, 645, 601), fill=BLUE)
    draw.text((650, 588), "drone A", fill=MUTED, font=F10)
    draw.ellipse((710, 591, 720, 601), fill=PURPLE)
    draw.text((725, 588), "drone B", fill=MUTED, font=F10)
    draw.line((789, 596, 805, 596), fill=ORANGE, width=4)
    draw.text((810, 588), "payload path", fill=MUTED, font=F10)
    draw.text((1094, 594), "seed 2002", fill="#91A0AF", font=F10)


def header(draw: ImageDraw.ImageDraw, progress: float) -> None:
    draw.text((30, 18), "FLIGHT PATHS OF THE BOUNDARY-DESIGNING DRONES", fill=INK, font=B27)
    draw.text((31, 53), "Actual cross-track trajectories • horizontal axis = normalized mission progress",
              fill=MUTED, font=F14)
    pill(draw, (1002, 21, 1169, 49), "V5 SIMULATOR DATA", BLUE, BLUE_BG, B10)
    draw.line((31, 91, 1169, 91), fill="#D4DEE8", width=5)
    draw.line((31, 91, 31 + 1138 * progress, 91), fill=GREEN, width=5)
    x = 31 + 1138 * progress
    draw.ellipse((x - 6, 85, x + 6, 97), fill=GREEN, outline=WHITE, width=2)


def footer(draw: ImageDraw.ImageDraw) -> None:
    draw.text((31, 648),
              "The halo represents the inferred/enacted Markov blanket; its transition to green tracks the measured causal gap.",
              fill=MUTED, font=F10)
    draw.text((929, 648), "Luca M. Possati • Designing the Boundary", fill=MUTED, font=F10)


def render(output: Path, frames_count: int = 96) -> None:
    world_i, agent_i = make_system(1000, "constitutive", "construct")
    result_i = run_mission(world_i, agent_i)
    world_c, team_c = make_collective_system(2002, "collective_constitutive", "assemble")
    result_c = run_collective_mission(world_c, team_c)
    if not result_i.success or not result_c.success:
        raise RuntimeError("Selected representative missions must succeed")

    frames: list[Image.Image] = []
    for k in range(frames_count):
        progress = k / max(frames_count - 1, 1)
        idx_i = int(round(progress * (len(result_i.records) - 1)))
        idx_c = int(round(progress * (len(result_c.records) - 1)))
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        header(draw, progress)
        individual_panel(draw, result_i.records, idx_i, k)
        collective_panel(draw, result_c.records, idx_c, k)
        footer(draw)
        frames.append(image)

    frames.extend([frames[-1].copy() for _ in range(16)])
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=92,
                   loop=0, optimize=True, disposal=2)
    print({"output": str(output), "frames": len(frames),
           "individual_success": result_i.success,
           "collective_success": result_c.success,
           "individual_changes": result_i.causal_changes,
           "collective_changes": result_c.causal_changes})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("assets/experiments_overview.gif"))
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    render(args.output, args.frames)


if __name__ == "__main__":
    main()
