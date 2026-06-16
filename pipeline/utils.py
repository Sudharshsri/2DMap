"""Shared utilities: room positioning logic used by stage4 and stage5."""

SIZE_HINT_DIMS = {
    "very_small": (2.0, 2.0),
    "small":      (3.0, 3.0),
    "medium":     (4.0, 4.0),
    "large":      (5.0, 5.0),
    "very_large": (6.0, 6.0),
}

def assign_room_positions(rooms: list, transitions: list) -> tuple[dict, dict]:
    """
    Return position_map {room_id: (x, y)} and heading_map {room_id: heading_deg}.

    Rooms are placed adjacently using a global coordinate system.
    The camera starts in the first room facing North (heading 0).
    As it goes through doors, the local door side (left/right/front/back)
    is converted to a global direction to place the next room, and the
    camera's heading rotates accordingly.
    """
    if not rooms:
        return {}, {}

    position_map: dict = {}
    heading_map: dict = {}
    
    start_id = rooms[0]["id"]
    position_map[start_id] = (0.0, 0.0)
    heading_map[start_id] = 0.0

    room_lookup = {r["id"]: r for r in rooms}

    # Adjacency list for transitions
    adj = {r["id"]: [] for r in rooms}
    for t in transitions:
        # Ignore invalid transitions missing from rooms
        if t.get("from_room") in adj:
            adj[t["from_room"]].append(t)

    # BFS to place all rooms
    queue = [start_id]
    visited = {start_id}

    while queue:
        curr = queue.pop(0)
        fx, fy = position_map[curr]
        fhding = heading_map[curr]
        fw = room_lookup[curr].get("width",  4.0)
        fh = room_lookup[curr].get("height", 4.0)

        for t in adj[curr]:
            to_id = t.get("to_room")
            if not to_id or to_id in position_map:
                continue

            local_side = t.get("door_position", "front")

            # Local to Global mapping
            rel_angle = {"front": 0, "right": 90, "back": 180, "left": 270}.get(local_side, 0)
            abs_heading = (fhding + rel_angle) % 360

            tw = room_lookup[to_id].get("width",  4.0)
            th = room_lookup[to_id].get("height", 4.0)

            # Place next room based on absolute direction
            if abs_heading == 0:       # North (+Y)
                tx = fx + (fw - tw) / 2
                ty = fy + fh
            elif abs_heading == 90:    # East (+X)
                tx = fx + fw
                ty = fy + (fh - th) / 2
            elif abs_heading == 180:   # South (-Y)
                tx = fx + (fw - tw) / 2
                ty = fy - th
            else:                      # West (-X) (abs_heading == 270)
                tx = fx - tw
                ty = fy + (fh - th) / 2

            position_map[to_id] = (round(tx, 2), round(ty, 2))
            heading_map[to_id] = abs_heading
            
            queue.append(to_id)
            visited.add(to_id)

    # Any rooms that didn't get placed via transitions go in a column on the right
    placed_xs = [v[0] for v in position_map.values()]
    x_stack = max(placed_xs) + 8.0 if placed_xs else 0.0
    y_stack = 0.0
    for room in rooms:
        rid = room["id"]
        if rid not in position_map:
            position_map[rid] = (round(x_stack, 2), round(y_stack, 2))
            heading_map[rid] = 0.0
            y_stack += room.get("height", 4.0) + 1.0

    return position_map, heading_map


def compute_camera_path(rooms: list, transitions: list, position_map: dict, heading_map: dict) -> list:
    """
    Build a camera_path list of {x, y, heading_deg, from_segment_id, to_segment_id}
    by walking room centres in transition order.
    """
    if not rooms:
        return []

    # ordered room list following transition chain
    ordered_ids: list[str] = []
    visited: set = set()

    def _chain(start_id: str):
        ordered_ids.append(start_id)
        visited.add(start_id)
        for t in transitions:
            if t.get("from_room") == start_id and t.get("to_room") not in visited:
                _chain(t["to_room"])

    _chain(rooms[0]["id"])

    # add any rooms not reached by transitions
    for r in rooms:
        if r["id"] not in visited:
            ordered_ids.append(r["id"])

    room_lookup = {r["id"]: r for r in rooms}
    path = []
    seg_counter = 0

    for i, rid in enumerate(ordered_ids):
        room = room_lookup.get(rid, {})
        x, y = position_map.get(rid, (0.0, 0.0))
        cx = round(x + room.get("width",  4.0) / 2, 2)
        cy = round(y + room.get("height", 4.0) / 2, 2)

        cam_heading = heading_map.get(rid, 0.0)
        draw_angle = (90 - cam_heading) % 360

        path.append({
            "x": cx,
            "y": cy,
            "heading_deg": draw_angle, 
            "from_segment_id": seg_counter,
            "to_segment_id": seg_counter + 1,
        })
        seg_counter += 1

    return path
