"""
Stage 5 — CAD rendering: DXF (ezdxf) + PNG (Matplotlib).

Converts the global floor-plan JSON into:
  • floor_plan.dxf  – CAD-ready drawing with layers for rooms, doors,
                      ghost rooms, camera path, and text labels.
  • floor_plan.png  – colour-coded raster image for quick review.

Ghost room rendering (v3)
--------------------------
Rooms with ghost=True represent spaces the camera observed through a doorway
but never entered. They are rendered distinctly:

PNG:
  - No fill (transparent interior)
  - Dashed border (linestyle="--", linewidth=1.5, edgecolor="#888888")
  - Diagonal hatching ("///") in light gray to indicate "empty/unknown"
  - Italic label: room type + "[?unvisited]"
  - alpha=0.35 so the floor grid shows through
  - Semi-transparent door arc on their wall side (orange dashed)

DXF:
  - Separate layer "GHOST_ROOMS" (color=8, gray)
  - Outline drawn with DASHED linetype
  - Label on LABELS layer suffixed with " [?]"
"""
import math
import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch

from pipeline.utils import assign_room_positions, compute_camera_path

# ── visual constants ─────────────────────────────────────────────────────────

_ROOM_COLORS = {
    "entrance":    "#FFD700",
    "corridor":    "#ADD8E6",
    "hallway":     "#ADD8E6",
    "living_room": "#90EE90",
    "bedroom":     "#FFB6C1",
    "kitchen":     "#FFA07A",
    "bathroom":    "#87CEEB",
    "office":      "#DDA0DD",
    "stairwell":   "#D3D3D3",
    "lobby":       "#F0E68C",
    "unknown":     "#E8E8E8",
}

_GHOST_HATCH_COLOR = "#AAAAAA"
_GHOST_EDGE_COLOR  = "#888888"
_GHOST_ALPHA       = 0.35

_DOOR_WIDTH     = 0.8    # metres
_WALL_THICKNESS = 0.15   # metres (visual only in DXF labels)


# ── public API ───────────────────────────────────────────────────────────────

def render_floor_plan(floor_plan: dict, dxf_path: str, png_path: str) -> None:
    """
    Render *floor_plan* dict to DXF and PNG files.

    *floor_plan* must contain keys: "rooms", "transitions", "camera_path".
    """
    rooms       = floor_plan.get("rooms", [])
    transitions = floor_plan.get("transitions", [])

    # Always recompute positions so coordinates are guaranteed consistent.
    # Include ghost transitions so ghost rooms are positioned correctly.
    position_map, heading_map = assign_room_positions(rooms, transitions)
    camera_path  = compute_camera_path(
        [r for r in rooms if not r.get("ghost", False)],
        [t for t in transitions if not t.get("ghost", False)],
        position_map, heading_map
    )

    if not rooms:
        print("  WARNING: no rooms in floor plan – output files will be empty stubs.")

    _render_dxf(rooms, camera_path, position_map, heading_map, dxf_path)
    _render_png(rooms, camera_path, position_map, heading_map, png_path)


# ── DXF ──────────────────────────────────────────────────────────────────────

def _render_dxf(rooms, camera_path, position_map, heading_map, out_path):
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()

    # Layers
    for name, color in [
        ("WALLS",          7),
        ("GHOST_ROOMS",    8),
        ("DOORS",          3),
        ("UNVISITED_DOORS",8),
        ("CAMERA_PATH",    1),
        ("LABELS",         2),
    ]:
        doc.layers.add(name, color=color)

    for room in rooms:
        rid   = room["id"]
        x, y  = position_map.get(rid, (0.0, 0.0))
        w, h  = room.get("width", 4.0), room.get("height", 4.0)
        is_ghost = room.get("ghost", False)

        layer = "GHOST_ROOMS" if is_ghost else "WALLS"

        msp.add_lwpolyline(
            [(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
            dxfattribs={"layer": layer, "closed": True},
        )

        # Room label
        suffix = " [?]" if is_ghost else ""
        label  = f"{room['type']}{suffix}  [{rid}]  ({room['size_hint']})"
        msp.add_text(
            label,
            dxfattribs={
                "layer":  "LABELS",
                "height": min(w, h) * 0.10,
                "insert": (x + w / 2, y + h / 2),
                "halign": 1,
            },
        ).set_placement(
            (x + w / 2, y + h / 2),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

    # Doors (traversed = solid DOORS; untraversed = dashed UNVISITED_DOORS)
    for room in rooms:
        rid  = room["id"]
        x, y = position_map.get(rid, (0.0, 0.0))
        w, h  = room.get("width", 4.0), room.get("height", 4.0)
        for door in room.get("door_locations", []):
            p1, p2 = _door_segment(x, y, w, h, door["side"], heading_map.get(rid, 0.0))
            if p1:
                layer = "DOORS" if door.get("traversed", True) else "UNVISITED_DOORS"
                msp.add_line(p1, p2, dxfattribs={"layer": layer})

    # Camera path + arrows
    if len(camera_path) >= 2:
        pts = [(p["x"], p["y"]) for p in camera_path]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "CAMERA_PATH"})

        for pt in camera_path:
            ang = math.radians(pt.get("heading_deg", 0.0))
            _draw_arrow_dxf(msp, pt["x"], pt["y"], ang, "CAMERA_PATH")

    doc.saveas(out_path)
    print(f"  DXF saved: {out_path}")


def _draw_arrow_dxf(msp, x, y, angle_rad, layer):
    """Draw a small directional arrow at (x, y) pointing in angle_rad."""
    length = 0.6
    head   = 0.18
    ax, ay = math.cos(angle_rad), math.sin(angle_rad)
    tx, ty = x + length * ax, y + length * ay
    msp.add_line((x, y), (tx, ty), dxfattribs={"layer": layer})
    for sign in (1, -1):
        side_a = angle_rad + sign * math.radians(150)
        msp.add_line(
            (tx, ty),
            (tx + head * math.cos(side_a), ty + head * math.sin(side_a)),
            dxfattribs={"layer": layer},
        )


# ── PNG ───────────────────────────────────────────────────────────────────────

def _render_png(rooms, camera_path, position_map, heading_map, out_path):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_aspect("equal")
    ax.set_facecolor("#F0F0F0")

    legend_patches = []
    ghost_legend_added = False
    traversed_door_legend_added = False
    unvisited_door_legend_added = False

    # ── Draw real rooms first, then ghost rooms on top ───────────────────────
    real_rooms  = [r for r in rooms if not r.get("ghost", False)]
    ghost_rooms = [r for r in rooms if r.get("ghost", False)]

    for room in real_rooms:
        _draw_real_room(ax, room, position_map, legend_patches)

    for room in ghost_rooms:
        _draw_ghost_room(ax, room, position_map)
        if not ghost_legend_added:
            legend_patches.append(
                mpatches.Patch(
                    facecolor="none", edgecolor=_GHOST_EDGE_COLOR,
                    hatch="///", alpha=_GHOST_ALPHA + 0.2,
                    label="Ghost room (unvisited, camera observed door only)",
                )
            )
            ghost_legend_added = True

    # ── Draw doors for all rooms ─────────────────────────────────────────────
    for room in rooms:
        rid  = room["id"]
        x, y = position_map.get(rid, (0.0, 0.0))
        w, h = room.get("width", 4.0), room.get("height", 4.0)

        for door in room.get("door_locations", []):
            p1, p2 = _door_segment(x, y, w, h, door["side"], heading_map.get(rid, 0.0))
            if not p1:
                continue

            if door.get("traversed", True):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="darkorange", linewidth=4,
                        solid_capstyle="round", zorder=6)
                if not traversed_door_legend_added:
                    legend_patches.append(
                        mlines.Line2D([0], [0], color="darkorange", linewidth=4,
                                      label="Door (camera traversed)")
                    )
                    traversed_door_legend_added = True
            else:
                # Untraversed door — dashed gray
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#808080", linewidth=2, linestyle="--",
                        solid_capstyle="round", zorder=6)
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                ax.text(mx, my, "?", ha="center", va="center",
                        fontsize=9, color="#555555", fontweight="bold", zorder=7)
                if not unvisited_door_legend_added:
                    legend_patches.append(
                        mlines.Line2D([0], [0], color="#808080", linewidth=2,
                                      linestyle="--",
                                      label="Door (unvisited — ghost room beyond)")
                    )
                    unvisited_door_legend_added = True

    # ── Camera path ──────────────────────────────────────────────────────────
    if len(camera_path) >= 2:
        xs = [p["x"] for p in camera_path]
        ys = [p["y"] for p in camera_path]
        ax.plot(xs, ys, "r-", linewidth=2.5, zorder=8)
        ax.plot(xs[0],  ys[0],  "go", markersize=12, zorder=9)
        ax.plot(xs[-1], ys[-1], "rs", markersize=12, zorder=9)

        for i in range(len(camera_path) - 1):
            cx, cy = camera_path[i]["x"],   camera_path[i]["y"]
            nx, ny = camera_path[i+1]["x"], camera_path[i+1]["y"]
            dx, dy = nx - cx, ny - cy
            dist   = math.hypot(dx, dy)
            if dist > 0.05:
                ax.annotate(
                    "", xy=(nx, ny), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    zorder=10,
                )

        legend_patches += [
            mpatches.Patch(color="green", label="Camera start"),
            mpatches.Patch(color="red",   label="Camera end / path"),
        ]

    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=8, framealpha=0.90)
    ax.autoscale()
    ax.margins(0.15)
    ax.set_title(
        "2D Floor Plan  —  generated from indoor video\n"
        "(solid rooms = camera visited  |  dashed rooms = observed through doorway only)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("X  (metres)")
    ax.set_ylabel("Y  (metres)")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {out_path}")


def _draw_real_room(ax, room, position_map, legend_patches):
    """Draw a fully visited (solid) room."""
    rid   = room["id"]
    x, y  = position_map.get(rid, (0.0, 0.0))
    w, h  = room.get("width", 4.0), room.get("height", 4.0)
    rtype = room.get("type", "unknown")
    color = _ROOM_COLORS.get(rtype, "#E8E8E8")

    rect = mpatches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor="black", facecolor=color, alpha=0.80,
        zorder=2,
    )
    ax.add_patch(rect)

    ax.text(
        x + w / 2, y + h / 2,
        f"{rtype}\n{rid}\n({room['size_hint']})",
        ha="center", va="center",
        fontsize=max(7, min(11, int(min(w, h) * 2.2))),
        fontweight="bold", zorder=3,
    )

    if not any(p.get_label() == rtype for p in legend_patches):
        legend_patches.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=rtype)
        )


def _draw_ghost_room(ax, room, position_map):
    """
    Draw a ghost room — dashed outline, hatched interior, italic label.
    Ghost rooms represent spaces the camera saw through a doorway but never entered.
    """
    rid   = room["id"]
    x, y  = position_map.get(rid, (0.0, 0.0))
    w, h  = room.get("width", 4.0), room.get("height", 4.0)
    rtype = room.get("type", "unknown")

    # Dashed border rectangle, no fill, with hatch
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        linewidth=1.5, linestyle="--",
        edgecolor=_GHOST_EDGE_COLOR,
        facecolor="white",
        alpha=_GHOST_ALPHA,
        hatch="///",
        zorder=4,
    )
    ax.add_patch(rect)

    # Italic label
    label = f"{rtype}\n[?unvisited]"
    ax.text(
        x + w / 2, y + h / 2,
        label,
        ha="center", va="center",
        fontsize=max(6, min(9, int(min(w, h) * 2.0))),
        fontstyle="italic",
        color="#555555",
        zorder=5,
    )

    # Subtle question-mark badge in corner
    ax.text(
        x + w - 0.2, y + h - 0.2, "?",
        ha="right", va="top",
        fontsize=14, color=_GHOST_EDGE_COLOR,
        fontweight="bold", zorder=5,
        alpha=0.7,
    )


# ── shared geometry ───────────────────────────────────────────────────────────

def _door_segment(x, y, w, h, local_side: str, room_heading: float):
    """Return two endpoints of a door gap on the requested wall side."""
    rel_angle = {"front": 0, "right": 90, "back": 180, "left": 270}.get(local_side, 0)
    abs_heading = (room_heading + rel_angle) % 360
    
    abs_side = "front"
    if abs_heading == 0: abs_side = "front"     # +Y
    elif abs_heading == 90: abs_side = "right"  # +X
    elif abs_heading == 180: abs_side = "back"  # -Y
    elif abs_heading == 270: abs_side = "left"  # -X

    cx, cy = x + w / 2, y + h / 2
    half   = _DOOR_WIDTH / 2
    if abs_side == "right":
        return (x+w, cy - half), (x+w, cy + half)
    if abs_side == "left":
        return (x,   cy - half), (x,   cy + half)
    if abs_side == "front":
        return (cx - half, y+h), (cx + half, y+h)
    if abs_side == "back":
        return (cx - half, y),   (cx + half, y)
    return None, None
