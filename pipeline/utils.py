"""Shared utilities: room positioning logic used by stage4 and stage5."""

SIZE_HINT_DIMS = {
    "very_small": (2.0, 2.0),
    "small":      (3.0, 3.0),
    "medium":     (4.0, 4.0),
    "large":      (5.0, 5.0),
    "very_large": (6.0, 6.0),
}

_DIRECTION_OFFSETS = {
    "right": (1,  0),
    "left":  (-1, 0),
    "front": (0,  1),
    "back":  (0, -1),
}


def assign_room_positions(rooms: list, transitions: list) -> dict:
    """
    Return {room_id: (x, y)} bottom-left corner for every room.

    Rooms are placed adjacently according to transition door_position:
      right  → next room is placed to the +x side
      left   → next room is placed to the -x side
      front  → next room is placed to the +y side
      back   → next room is placed to the -y side
    """
    if not rooms:
        return {}

    position_map: dict = {}
    position_map[rooms[0]["id"]] = (0.0, 0.0)

    room_lookup = {r["id"]: r for r in rooms}

    for t in transitions:
        from_id  = t.get("from_room")
        to_id    = t.get("to_room")
        door_pos = t.get("door_position", "front")

        if from_id not in position_map or to_id in position_map:
            continue

        from_room = room_lookup.get(from_id, {})
        to_room   = room_lookup.get(to_id,   {})

        fx, fy = position_map[from_id]
        fw = from_room.get("width",  4.0)
        fh = from_room.get("height", 4.0)
        tw = to_room.get("width",    4.0)
        th = to_room.get("height",   4.0)

        dx, dy = _DIRECTION_OFFSETS.get(door_pos, (0, 1))

        if dx == 1:       # right
            tx = fx + fw
            ty = fy + (fh - th) / 2
        elif dx == -1:    # left
            tx = fx - tw
            ty = fy + (fh - th) / 2
        elif dy == 1:     # front
            tx = fx + (fw - tw) / 2
            ty = fy + fh
        else:             # back
            tx = fx + (fw - tw) / 2
            ty = fy - th

        position_map[to_id] = (round(tx, 2), round(ty, 2))

    # Any rooms that didn't get placed via transitions go in a column on the right
    placed_xs = [v[0] for v in position_map.values()]
    x_stack = max(placed_xs) + 8.0 if placed_xs else 0.0
    y_stack = 0.0
    for room in rooms:
        if room["id"] not in position_map:
            position_map[room["id"]] = (round(x_stack, 2), round(y_stack, 2))
            y_stack += room.get("height", 4.0) + 1.0

    return position_map


def compute_camera_path(rooms: list, transitions: list, position_map: dict,
                        segments: list | None = None) -> list:
    """
    Build a camera_path list of {x, y, heading_deg, from_segment_id, to_segment_id}
    by walking room centres in transition order.
    """
    if not rooms:
        return []

    _HEADING = {"right": 90.0, "left": -90.0, "front": 0.0, "back": 180.0}

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

        heading = 0.0
        if i > 0:
            prev_id = ordered_ids[i - 1]
            for t in transitions:
                if t.get("from_room") == prev_id and t.get("to_room") == rid:
                    heading = _HEADING.get(t.get("door_position", "front"), 0.0)
                    break

        path.append({
            "x": cx,
            "y": cy,
            "heading_deg": heading,
            "from_segment_id": seg_counter,
            "to_segment_id": seg_counter + 1,
        })
        seg_counter += 1

    return path
