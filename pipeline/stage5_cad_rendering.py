"""
Stage 5 — CAD rendering: DXF (ezdxf) + PNG (Matplotlib).

Converts the global floor-plan JSON into:
  • floor_plan.dxf  – CAD-ready drawing with layers for rooms, doors,
                      camera path, and text labels.
  • floor_plan.png  – colour-coded raster image for quick review.
"""
import math
import ezdxf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

_DOOR_WIDTH     = 0.8    # metres
_WALL_THICKNESS = 0.15   # metres (visual only in DXF labels)


# ── public API ───────────────────────────────────────────────────────────────

def render_floor_plan(floor_plan: dict, dxf_path: str, png_path: str) -> None:
    """
    Render *floor_plan* dict to DXF and PNG files.

    *floor_plan* must contain keys: "rooms", "transitions", "camera_path".
    """
    rooms        = floor_plan.get("rooms", [])
    transitions  = floor_plan.get("transitions", [])

    # Always recompute positions so coordinates are guaranteed consistent
    position_map = assign_room_positions(rooms, transitions)
    camera_path  = compute_camera_path(rooms, transitions, position_map)

    if not rooms:
        print("  WARNING: no rooms in floor plan – output files will be empty stubs.")

    _render_dxf(rooms, camera_path, position_map, dxf_path)
    _render_png(rooms, camera_path, position_map, png_path)


# ── DXF ──────────────────────────────────────────────────────────────────────

def _render_dxf(rooms, camera_path, position_map, out_path):
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()

    # Layers
    for name, color in [("WALLS", 7), ("DOORS", 3),
                        ("CAMERA_PATH", 1), ("LABELS", 2)]:
        doc.layers.add(name, color=color)

    room_lookup = {r["id"]: r for r in rooms}

    # Rooms
    for room in rooms:
        rid  = room["id"]
        x, y = position_map.get(rid, (0.0, 0.0))
        w, h  = room.get("width", 4.0), room.get("height", 4.0)

        msp.add_lwpolyline(
            [(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
            dxfattribs={"layer": "WALLS", "closed": True},
        )

        # Room label (two-line: type + ID + size)
        label = f"{room['type']}  [{rid}]  ({room['size_hint']})"
        msp.add_text(
            label,
            dxfattribs={
                "layer":  "LABELS",
                "height": min(w, h) * 0.12,
                "insert": (x + w / 2, y + h / 2),
                "halign": 1,   # MIDDLE
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
            p1, p2 = _door_segment(x, y, w, h, door["side"])
            if p1:
                msp.add_line(p1, p2, dxfattribs={"layer": "DOORS"})

    # Camera path + arrows
    if len(camera_path) >= 2:
        pts = [(p["x"], p["y"]) for p in camera_path]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "CAMERA_PATH"})

        for i, pt in enumerate(camera_path):
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

def _render_png(rooms, camera_path, position_map, out_path):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_aspect("equal")
    ax.set_facecolor("#F0F0F0")

    legend_patches = []

    for room in rooms:
        rid    = room["id"]
        x, y   = position_map.get(rid, (0.0, 0.0))
        w, h   = room.get("width", 4.0), room.get("height", 4.0)
        rtype  = room.get("type", "unknown")
        color  = _ROOM_COLORS.get(rtype, "#E8E8E8")

        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="black", facecolor=color, alpha=0.75,
            zorder=2,
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x + w / 2, y + h / 2,
            f"{rtype}\n{rid}\n({room['size_hint']})",
            ha="center", va="center",
            fontsize=max(7, min(11, int(min(w, h) * 2.2))),
            fontweight="bold", zorder=3,
        )

        # Legend entry (unique)
        if not any(p.get_label() == rtype for p in legend_patches):
            legend_patches.append(
                mpatches.Patch(facecolor=color, edgecolor="black", label=rtype)
            )

        # Doors
        for door in room.get("door_locations", []):
            p1, p2 = _door_segment(x, y, w, h, door["side"])
            if p1:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="darkorange", linewidth=4,
                        solid_capstyle="round", zorder=4)

    # Camera path
    if len(camera_path) >= 2:
        xs = [p["x"] for p in camera_path]
        ys = [p["y"] for p in camera_path]
        ax.plot(xs, ys, "r-", linewidth=2.5, zorder=5)
        ax.plot(xs[0],  ys[0],  "go", markersize=12, zorder=6)
        ax.plot(xs[-1], ys[-1], "rs", markersize=12, zorder=6)

        for i in range(len(camera_path) - 1):
            cx, cy   = camera_path[i]["x"],   camera_path[i]["y"]
            nx, ny   = camera_path[i+1]["x"], camera_path[i+1]["y"]
            dx, dy   = nx - cx, ny - cy
            dist     = math.hypot(dx, dy)
            if dist > 0.05:
                ax.annotate(
                    "", xy=(nx, ny), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2),
                    zorder=7,
                )

        legend_patches += [
            mpatches.Patch(color="green",  label="Camera start"),
            mpatches.Patch(color="red",    label="Camera end"),
            mpatches.Patch(color="darkorange", label="Door"),
        ]

    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=9, framealpha=0.85)
    ax.autoscale()
    ax.margins(0.12)
    ax.set_title("2D Floor Plan  —  generated from indoor video",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("X  (metres)")
    ax.set_ylabel("Y  (metres)")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved: {out_path}")


# ── shared geometry ───────────────────────────────────────────────────────────

def _door_segment(x, y, w, h, side: str):
    """Return two endpoints of a door gap on the requested wall side."""
    cx, cy = x + w / 2, y + h / 2
    half   = _DOOR_WIDTH / 2
    if side == "right":
        return (x+w, cy - half), (x+w, cy + half)
    if side == "left":
        return (x,   cy - half), (x,   cy + half)
    if side == "front":
        return (cx - half, y+h), (cx + half, y+h)
    if side == "back":
        return (cx - half, y),   (cx + half, y)
    return None, None
