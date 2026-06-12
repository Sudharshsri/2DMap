import ezdxf
import json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc

def generate_floor_plan(json_path, output_dxf, output_png):
    with open(json_path) as f:
        data = json.load(f)

    _build_dxf(data, output_dxf)
    _build_png(data, output_png)

# ── DXF ──────────────────────────────────────────────────────────────────────
def _build_dxf(data, output_dxf):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # millimeters

    # Layers per IS 962 Table 9.2
    doc.layers.add("BORDER",   color=7, lineweight=50)   # Continuous Thick 0.5mm
    doc.layers.add("WALLS",    color=7, lineweight=50)   # Continuous Thick 0.5mm
    doc.layers.add("DOORS",    color=1, lineweight=25)   # Continuous Thin 0.25mm
    doc.layers.add("WINDOWS",  color=5, lineweight=25)   # Continuous Thin 0.25mm
    doc.layers.add("LABELS",   color=7, lineweight=18)   # Thin text layer
    doc.layers.add("TITLE",    color=7, lineweight=25)

    msp = doc.modelspace()
    rooms = {r["id"]: r for r in data.get("rooms", [])}

    # A4 landscape outer border
    msp.add_lwpolyline([(5,5),(292,5),(292,205),(5,205)],
                       close=True, dxfattribs={"layer":"BORDER"})
    # Inner drawing frame
    msp.add_lwpolyline([(15,32),(282,32),(282,195),(15,195)],
                       close=True, dxfattribs={"layer":"BORDER"})

    # Rooms
    for room in data.get("rooms", []):
        x, y, w, h = room["x"], room["y"], room["width"], room["height"]
        msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
                           close=True, dxfattribs={"layer":"WALLS"})
        # IS 962 Sec 10: room label centered, uppercase, 3.5mm
        msp.add_text(room.get("name", "ROOM").upper(),
                     dxfattribs={"layer":"LABELS","height":3.5,"insert":(x+w/2, y+h/2)})

    # Doors (IS 962 Sec 11: arc + panel line)
    for door in data.get("doors", []):
        if door.get("room_id"):
            room = rooms.get(door["room_id"])
            if not room: continue
            x,y,w,h = room["x"],room["y"],room["width"],room["height"]
            r,dw = door["position_ratio"], door["width"]
            wall, swing = door["wall"], door.get("swing","left")
    
            if wall == "bottom":
                hx,hy = x+w*r, y
                msp.add_line((hx,hy),(hx,hy+dw),   dxfattribs={"layer":"DOORS"})
                a1,a2 = (0,90) if swing=="right" else (90,180)
            elif wall == "top":
                hx,hy = x+w*r, y+h
                msp.add_line((hx,hy),(hx,hy-dw),   dxfattribs={"layer":"DOORS"})
                a1,a2 = (270,360) if swing=="right" else (180,270)
            elif wall == "left":
                hx,hy = x, y+h*r
                msp.add_line((hx,hy),(hx+dw,hy),   dxfattribs={"layer":"DOORS"})
                a1,a2 = (0,90) if swing=="right" else (270,360)
            elif wall == "right":
                hx,hy = x+w, y+h*r
                msp.add_line((hx,hy),(hx-dw,hy),   dxfattribs={"layer":"DOORS"})
                a1,a2 = (180,270) if swing=="right" else (90,180)
            else: continue
            msp.add_arc((hx,hy), dw, a1, a2, dxfattribs={"layer":"DOORS"})
        elif "x" in door and "y" in door:
            x, y, w, h = door["x"], door["y"], door.get("width", 10), door.get("height", 10)
            msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
                               close=True, dxfattribs={"layer":"DOORS"})

    # Windows (IS 962 Sec 11: 3 parallel lines across wall opening)
    for win in data.get("windows", []):
        if win.get("room_id"):
            room = rooms.get(win["room_id"])
            if not room: continue
            x,y,w,h = room["x"],room["y"],room["width"],room["height"]
            r,ww = win["position_ratio"], win["width"]
            wall, off = win["wall"], 1.5
    
            if wall in ("bottom","top"):
                wx = x+w*r
                wy = y if wall=="bottom" else y+h
                for dy in (-off,0,off):
                    msp.add_line((wx,wy+dy),(wx+ww,wy+dy), dxfattribs={"layer":"WINDOWS"})
            elif wall in ("left","right"):
                wx = x if wall=="left" else x+w
                wy = y+h*r
                for dx in (-off,0,off):
                    msp.add_line((wx+dx,wy),(wx+dx,wy+ww), dxfattribs={"layer":"WINDOWS"})
        elif "x" in win and "y" in win:
            x, y, w, h = win["x"], win["y"], win.get("width", 10), win.get("height", 10)
            msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],
                               close=True, dxfattribs={"layer":"WINDOWS"})

    # Stairs (architectural: outline + horizontal step lines + direction arrow)
    for stair in data.get("stairs", []):
        sx, sy = stair.get("x", 0), stair.get("y", 0)
        sw, sh = stair.get("width", 22), stair.get("height", 16)
        msp.add_lwpolyline([(sx, sy), (sx+sw, sy), (sx+sw, sy+sh), (sx, sy+sh)],
                           close=True, dxfattribs={"layer": "WALLS"})
        n_steps = max(3, int(sh / 3))
        step_h = sh / n_steps
        for i in range(1, n_steps):
            y_line = sy + i * step_h
            msp.add_line((sx, y_line), (sx + sw, y_line), dxfattribs={"layer": "WALLS"})
        # Direction arrow (up arrow from bottom to top of stair box)
        mid_x = sx + sw / 2
        msp.add_line((mid_x, sy + 1), (mid_x, sy + sh - 1), dxfattribs={"layer": "LABELS"})

    # Title block (IS 962 Sec 10)
    meta = data.get("metadata",{})
    msp.add_lwpolyline([(15,5),(282,5),(282,30),(15,30)],
                       close=True, dxfattribs={"layer":"TITLE"})
    msp.add_text(meta.get("title","FLOOR PLAN").upper(),
                 dxfattribs={"layer":"TITLE","height":7,"insert":(148,20)})
    msp.add_text(f"Scale: {meta.get('scale','NTS')}   |   Sheet: A4   |   IS 962",
                 dxfattribs={"layer":"TITLE","height":2.5,"insert":(148,14)})

    os.makedirs(os.path.dirname(output_dxf), exist_ok=True)
    doc.saveas(output_dxf)
    print(f"  DXF saved: {output_dxf}")

# ── PNG Preview ───────────────────────────────────────────────────────────────
def _build_png(data, output_png):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape inches
    ax.set_xlim(0,297); ax.set_ylim(0,210)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")

    rooms = {r["id"]: r for r in data.get("rooms",[])}
    palette = ["#E8F4F8","#F8F0E8","#F0F8E8","#F8E8F4","#F0EAF8"]

    # Border & frame
    ax.add_patch(mpatches.Rectangle((5,5),287,200,lw=1.5,ec="black",fc="white"))
    ax.add_patch(mpatches.Rectangle((15,32),267,163,lw=1,ec="black",fc="#FAFAFA"))

    # Rooms
    for i,room in enumerate(data.get("rooms",[])):
        x,y,w,h = room["x"],room["y"],room["width"],room["height"]
        ax.add_patch(mpatches.Rectangle((x,y),w,h,lw=2,ec="black",
                                         fc=palette[i%len(palette)], zorder=2))
        ax.text(x+w/2, y+h/2, room.get("name", "ROOM").upper(),
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#222", zorder=7)

    # Doors
    for door in data.get("doors",[]):
        if door.get("room_id"):
            room = rooms.get(door["room_id"])
            if not room: continue
            x,y,w,h = room["x"],room["y"],room["width"],room["height"]
            r,dw = door["position_ratio"], door["width"]
            wall = door["wall"]
            swing = door.get("swing", "left")
            if wall=="bottom":
                hx,hy = x+w*r,y
                a1,a2 = (0,90) if swing=="right" else (90,180)
            elif wall=="top":
                hx,hy = x+w*r,y+h
                a1,a2 = (270,360) if swing=="right" else (180,270)
            elif wall=="left":
                hx,hy = x,y+h*r
                a1,a2 = (0,90) if swing=="right" else (270,360)
            elif wall=="right":
                hx,hy = x+w,y+h*r
                a1,a2 = (180,270) if swing=="right" else (90,180)
            else: continue
            ax.add_patch(Arc((hx,hy),2*dw,2*dw,angle=0,
                             theta1=a1,theta2=a2,color="#3366CC",lw=1.5))
            if wall in ("bottom","top"):
                ax.plot([hx,hx],[hy,hy+(dw if wall=="bottom" else -dw)],
                        color="#3366CC",lw=1.5)
            else:
                ax.plot([hx,hx+(dw if wall=="left" else -dw)],[hy,hy],
                        color="#3366CC",lw=1.5)
        elif "x" in door and "y" in door:
            x,y,w,h = door["x"],door["y"],door.get("width", 10),door.get("height", 10)
            ax.add_patch(mpatches.Rectangle((x,y),w,h,lw=1.5,ec="#3366CC",fc="none"))

    # Windows
    for win in data.get("windows",[]):
        if win.get("room_id"):
            room = rooms.get(win["room_id"])
            if not room: continue
            x,y,w,h = room["x"],room["y"],room["width"],room["height"]
            r,ww,off = win["position_ratio"],win["width"],1.5
            wall = win["wall"]
            if wall in ("bottom","top"):
                wx=x+w*r; wy=y if wall=="bottom" else y+h
                for dy in(-off,0,off):
                    ax.plot([wx,wx+ww],[wy+dy,wy+dy],color="#228B22",lw=1.5)
            elif wall in ("left","right"):
                wx=x if wall=="left" else x+w; wy=y+h*r
                for dx in(-off,0,off):
                    ax.plot([wx+dx,wx+dx],[wy,wy+ww],color="#228B22",lw=1.5)
        elif "x" in win and "y" in win:
            x,y,w,h = win["x"],win["y"],win.get("width", 10),win.get("height", 10)
            ax.add_patch(mpatches.Rectangle((x,y),w,h,lw=1.5,ec="#228B22",fc="none"))

    # Stairs (outline + horizontal step lines inside + label)
    for stair in data.get("stairs", []):
        sx, sy = stair.get("x", 0), stair.get("y", 0)
        sw, sh = stair.get("width", 22), stair.get("height", 16)
        ax.add_patch(mpatches.Rectangle((sx, sy), sw, sh,
                                        lw=1.5, ec="#333333", fc="#D8D8D8", zorder=3))
        n_steps = max(3, int(sh / 3))
        step_h = sh / n_steps
        for i in range(1, n_steps):
            y_line = sy + i * step_h
            ax.plot([sx, sx + sw], [y_line, y_line], color="#555555", lw=0.8, zorder=4)
        ax.text(sx + sw / 2, sy + sh / 2, "STAIRS",
                ha="center", va="center", fontsize=4.5, color="#222", fontweight="bold", zorder=5)

    # Title block
    meta = data.get("metadata",{})
    ax.add_patch(mpatches.Rectangle((15,5),267,25,lw=1,ec="black",fc="white"))
    ax.text(148,22,meta.get("title","FLOOR PLAN").upper(),
            ha="center",va="center",fontsize=13,fontweight="bold")
    ax.text(148,14,f"Scale: {meta.get('scale','NTS')}   |   Sheet: A4   |   IS 962",
            ha="center",va="center",fontsize=7,color="#555")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  PNG saved: {output_png}")

if __name__ == "__main__":
    import os
    import sys
    
    if not os.path.exists("output/floor_plan.json"):
        print("Error: output/floor_plan.json not found. Please run step2b_generate_json.py first.")
        sys.exit(1)
        
    # Generate the map from the existing JSON
    generate_floor_plan("output/floor_plan.json",
                        "output/floor_plan.dxf",
                        "output/floor_plan.png")
                        
    print("\n[OK] DXF and PNG floor plans saved in output/ directory")