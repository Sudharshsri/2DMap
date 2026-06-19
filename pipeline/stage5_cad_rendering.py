"""
Stage 5 — CAD rendering: DXF (ezdxf) + PNG (Matplotlib).

Changes from v3 (proximity-aware room labeling)
------------------------------------------------
Room name labels are now ONLY shown for rooms the camera was physically
inside or very close to. This prevents labels from cluttering the map for
rooms the camera merely walked past or observed distantly.

Labeling rules:
  INSIDE   : any camera path waypoint falls within the room's bounding box
             (with a small 0.5 m margin to account for waypoint at room centre)
  NEAR     : any camera path waypoint is within max(width, height) * 1.2 of
             the room's centre
  → Full label shown: room_type + room_id + size_hint (as before)
  → Distant room:     only a faint italic room_type initial (first letter)
                      so the user can still identify the room type at a glance

Ghost rooms always show their "?" label (camera never entered).

All other rendering (ghost outlines, doors, camera path) is unchanged from v3.
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

# Proximity multiplier: camera within (max_dim * factor) of room centre = "near"
_PROXIMITY_FACTOR = 1.2


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

    # Compute proximity set — rooms the camera was inside or near
    camera_near_set = _camera_proximity_set(camera_path, rooms, position_map)

    _render_dxf(rooms, camera_path, position_map, heading_map,
                camera_near_set, dxf_path)
    _render_png(rooms, camera_path, position_map, heading_map,
                camera_near_set, png_path)


# ── proximity helper ─────────────────────────────────────────────────────────

def _camera_proximity_set(camera_path: list, rooms: list,
                           position_map: dict) -> set:
    """
    Return a set of room IDs for which at least one camera waypoint is either:
      - INSIDE the room bounding box (expanded by 0.5 m margin), OR
      - NEAR the room centre (within max(w,h) * _PROXIMITY_FACTOR)

    Ghost rooms are never added (they are handled separately in the renderer).
    """
    near: set = set()
    margin = 0.5  # metres — expand bbox to catch waypoints just at the edge

    for room in rooms:
        if room.get("ghost", False):
            continue
        rid   = room["id"]
        x, y  = position_map.get(rid, (0.0, 0.0))
        w, h  = room.get("width", 4.0), room.get("height", 4.0)
        cx    = x + w / 2
        cy    = y + h / 2
        prox  = max(w, h) * _PROXIMITY_FACTOR

        for wp in camera_path:
            wx, wy = wp["x"], wp["y"]

            # Inside check (with margin)
            inside = (
                (x - margin) <= wx <= (x + w + margin) and
                (y - margin) <= wy <= (y + h + margin)
            )
            # Proximity check
            dist = math.hypot(wx - cx, wy - cy)
            if inside or dist <= prox:
                near.add(rid)
                break  # one waypoint is enough

    return near


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _get_straight_segments(camera_path: list) -> list:
    """Combine collinear waypoints into straight segments for distance labeling."""
    if len(camera_path) < 2:
        return []

    segments = []
    current_segment = {"points": [camera_path[0]], "dist": 0.0, "angle": None}

    for i in range(len(camera_path) - 1):
        p1 = camera_path[i]
        p2 = camera_path[i+1]
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            continue
        
        angle = math.atan2(dy, dx)
        
        if current_segment["angle"] is None:
            current_segment["angle"] = angle
            current_segment["points"].append(p2)
            current_segment["dist"] += dist
        else:
            angle_diff = abs(angle - current_segment["angle"])
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            # If direction change is less than 20 degrees, treat it as straight
            if angle_diff < math.radians(20):
                current_segment["points"].append(p2)
                current_segment["dist"] += dist
            else:
                # Direction changed — finalize current and start new
                segments.append(current_segment)
                current_segment = {"points": [p1, p2], "dist": dist, "angle": angle}

    if current_segment["angle"] is not None:
        segments.append(current_segment)

    # Compute midpoints
    for seg in segments:
        pts = seg["points"]
        seg["mid_x"] = (pts[0]["x"] + pts[-1]["x"]) / 2
        seg["mid_y"] = (pts[0]["y"] + pts[-1]["y"]) / 2

    return segments


# ── DXF ──────────────────────────────────────────────────────────────────────

def _render_dxf(rooms, camera_path, position_map, heading_map,
                camera_near_set: set, out_path: str):
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
        ("LABELS_DISTANT", 9),
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

        # Room label — full for near/inside rooms, abbreviated for distant ones
        if is_ghost:
            label = f"{room['type']} [?]  [{rid}]"
        elif rid in camera_near_set:
            label = f"{room['type']}  [{rid}]  ({room['size_hint']})"
            layer_lbl = "LABELS"
        else:
            label = f"{room['type'][0].upper()}?"
            layer_lbl = "LABELS_DISTANT"

        label_layer = "LABELS" if is_ghost else layer_lbl
        msp.add_text(
            label,
            dxfattribs={
                "layer":  label_layer,
                "height": min(w, h) * 0.10,
                "insert": (x + w / 2, y + h / 2),
                "halign": 1,
            },
        ).set_placement(
            (x + w / 2, y + h / 2),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

    # Doors
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

        # Label distances for straight line segments
        straight_segments = _get_straight_segments(camera_path)
        for seg in straight_segments:
            if seg["dist"] >= 0.5:
                # Place label slightly offset perpendicular to the path
                perp_angle = seg["angle"] + math.pi / 2
                off_x = seg["mid_x"] + math.cos(perp_angle) * 0.4
                off_y = seg["mid_y"] + math.sin(perp_angle) * 0.4
                
                msp.add_text(
                    f"{seg['dist']:.1f}m",
                    dxfattribs={
                        "layer": "LABELS",
                        "height": 0.25,
                        "insert": (off_x, off_y),
                    }
                ).set_placement(
                    (off_x, off_y),
                    align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
                )

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

def _render_png(rooms, camera_path, position_map, heading_map,
                camera_near_set: set, out_path: str):
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
        camera_near = room["id"] in camera_near_set
        _draw_real_room(ax, room, position_map, legend_patches, camera_near)

    for room in ghost_rooms:
        _draw_ghost_room(ax, room, position_map)
        if not ghost_legend_added:
            legend_patches.append(
                mpatches.Patch(
                    facecolor="none", edgecolor=_GHOST_EDGE_COLOR,
                    hatch="///", alpha=_GHOST_ALPHA + 0.2,
                    label="Ghost room (unvisited — camera observed door only)",
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

        # Draw distance labels
        straight_segments = _get_straight_segments(camera_path)
        for seg in straight_segments:
            if seg["dist"] >= 0.5:
                # Place label slightly offset perpendicular to the path
                perp_angle = seg["angle"] + math.pi / 2
                off_x = seg["mid_x"] + math.cos(perp_angle) * 0.5
                off_y = seg["mid_y"] + math.sin(perp_angle) * 0.5
                ax.text(off_x, off_y, f"{seg['dist']:.1f}m", color="darkred", fontsize=10, 
                        fontweight="bold", ha="center", va="center", zorder=11,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.8))

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
        "(solid rooms = camera visited  |  dashed rooms = observed through doorway only)\n"
        "(full label = camera was inside/near  |  letter only = camera was distant)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("X  (metres)")
    ax.set_ylabel("Y  (metres)")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {out_path}")


def _draw_real_room(ax, room, position_map, legend_patches, camera_near: bool = True):
    """
    Draw a fully visited (solid) room.
    camera_near=True  → draw full label (room_type + id + size)
    camera_near=False → draw only a faint italic initial so the room is
                        still identifiable but doesn't crowd the map.
    """
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

    if camera_near:
        # Full label
        ax.text(
            x + w / 2, y + h / 2,
            f"{rtype}\n{rid}\n({room['size_hint']})",
            ha="center", va="center",
            fontsize=max(7, min(11, int(min(w, h) * 2.2))),
            fontweight="bold", zorder=3,
        )
    else:
        # Distant room — faint italic room type initial
        ax.text(
            x + w / 2, y + h / 2,
            rtype[0].upper() if rtype else "?",
            ha="center", va="center",
            fontsize=max(10, min(16, int(min(w, h) * 3.0))),
            fontstyle="italic",
            color="#999999",
            alpha=0.6,
            zorder=3,
        )

    if not any(p.get_label() == rtype for p in legend_patches):
        legend_patches.append(
            mpatches.Patch(facecolor=color, edgecolor="black", label=rtype)
        )


def _draw_ghost_room(ax, room, position_map):
    """
    Draw a ghost room — dashed outline, hatched interior, italic label.
    Ghost rooms represent spaces the camera saw through a doorway but never entered.
    The dashed outline is always drawn, even if the camera only glimpsed a door.
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

    # Italic label — always shown for ghost rooms
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
    if abs_heading == 0:   abs_side = "front"
    elif abs_heading == 90:  abs_side = "right"
    elif abs_heading == 180: abs_side = "back"
    elif abs_heading == 270: abs_side = "left"

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
