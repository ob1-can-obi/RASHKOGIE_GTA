"""
game_state_reader.py
====================
Reads the GTA5 state stream from the Named Pipe and exposes it as a
thread-safe queue. Import GameStateReader and call .start() before your
training loop.

Usage
-----
    from game_state_reader import GameStateReader, FIELDS

    reader = GameStateReader(fields=None)   # None = all scalar fields
    reader.start()

    while True:
        state = reader.get()                # blocks until next frame
        ego = state.to_tensor()             # selected scalar ego fields
        world = flatten_world_tensor(state) # fixed-size mixed-entity tensor
"""

import json
import logging
import logging.handlers
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


def _build_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "game_state_reader.log")

    logger = logging.getLogger("gta5_reader")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _build_logger()


def _find_next_saved_frame_number(stats_dir: str) -> int:
    """
    Find the next available frame number so saved sessions append instead of
    overwriting older frames.
    """

    if not os.path.isdir(stats_dir):
        return 1

    max_frame_num = 0

    for filename in os.listdir(stats_dir):
        if not filename.startswith("frame_"):
            continue
        if not filename.endswith(".json"):
            continue

        frame_text = filename[len("frame_") : -len(".json")]
        if not frame_text.isdigit():
            continue

        frame_num = int(frame_text)
        if frame_num > max_frame_num:
            max_frame_num = frame_num

    return max_frame_num + 1


FIELDS: Dict[str, tuple] = {
    # Timing
    "ts": ("game tick timestamp (ms)", int, "monotonic"),

    # Player position & motion
    "px": ("player world X", float, "metres"),
    "py": ("player world Y", float, "metres"),
    "pz": ("player world Z", float, "metres"),
    "vx": ("player velocity X", float, "m/s"),
    "vy": ("player velocity Y", float, "m/s"),
    "vz": ("player velocity Z", float, "m/s"),
    "rx": ("player rotation pitch", float, "degrees"),
    "ry": ("player rotation roll", float, "degrees"),
    "rz": ("player rotation yaw/heading", float, "degrees"),

    # Player vitals
    "hp": ("player health", float, "0-200, 100=full"),
    "hp_max": ("player max health", float, "usually 200"),
    "armor": ("player armour", float, "0-100"),
    "wanted": ("police wanted level", int, "0-5"),
    "dead": ("player is dead", bool, ""),

    # World / environment
    "clock_h": ("in-game hour", int, "0-23"),
    "clock_m": ("in-game minute", int, "0-59"),
    "clock_s": ("in-game second", int, "0-59"),
    "weather": ("weather type enum", int, "0=EXTRASUNNY...15=HALLOWEEN"),
    "rain": ("rainfall intensity", float, "0.0-1.0"),
    "wind_speed": ("wind speed", float, "m/s"),
    "wind_x": ("wind direction X component", float, "unit vector"),
    "wind_y": ("wind direction Y component", float, "unit vector"),

    # Ego vehicle state
    "v_speed": ("vehicle speed", float, "m/s"),
    "v_throttle": ("throttle input", float, "0.0-1.0"),
    "v_brake": ("brake input", float, "0.0-1.0"),
    "v_steer": ("steering input", float, "-1.0..1.0"),
    "u_throttle": ("raw player throttle input", float, "0.0-1.0"),
    "u_brake": ("raw player brake input", float, "0.0-1.0"),
    "u_steer": ("raw player steering input", float, "-1.0..1.0"),
    "u_handbrake": ("raw player handbrake input", bool, ""),
    "u_horn": ("raw player horn input", bool, ""),
    "u_look_behind": ("raw player look-behind input", bool, ""),
    "u_duck": ("raw player duck input", bool, ""),
    "agent_active": ("automation command currently being applied", bool, ""),
    "agent_throttle": ("latest automation throttle command", float, "0.0-1.0"),
    "agent_brake": ("latest automation brake command", float, "0.0-1.0"),
    "agent_steer": ("latest automation steering command", float, "-1.0..1.0"),
    "agent_handbrake": ("latest automation handbrake command", bool, ""),
    "agent_age_ms": ("age of latest automation command", int, "milliseconds"),
    "v_rpm": ("engine RPM", float, "0.0-1.0 normalized"),
    "v_gear": ("current gear", int, "0=reverse,1=neutral,2+=drive"),
    "v_max_gear": ("highest gear available", int, ""),
    "v_steer_angle": ("actual wheel steering angle", float, "degrees"),
    "v_clutch": ("clutch engagement", float, "0.0-1.0"),
    "v_wheel_speed": ("drive wheel rotational speed", float, "m/s"),
    "v_accel": ("current acceleration", float, ""),
    "v_engine_hp": ("engine health", float, "0-1000"),
    "v_body_hp": ("body/chassis health", float, "0-1000"),
    "v_engine_on": ("engine is running", bool, ""),
    "v_turbo": ("turbo pressure", float, "0.0-1.0"),
    "v_fuel": ("fuel level", float, "0.0-1.0"),
    "v_model": ("vehicle model hash", int, "joaat hash"),
    "v_class": ("vehicle class enum", int, "0=compact..21=train"),
    "v_ax": ("local accel X (lateral)", float, "m/s²"),
    "v_ay": ("local accel Y (forward/back)", float, "m/s²"),
    "v_az": ("local accel Z (vertical)", float, "m/s²"),
    "v_heading": ("vehicle heading", float, "degrees, 0=north clockwise"),
    "v_fwd_x": ("forward unit vector X (world)", float, ""),
    "v_fwd_y": ("forward unit vector Y (world)", float, ""),
    "v_right_x": ("right unit vector X (world)", float, ""),
    "v_right_y": ("right unit vector Y (world)", float, ""),
    "v_dim_w": ("vehicle width", float, "metres"),
    "v_dim_l": ("vehicle length", float, "metres"),
    "v_dim_h": ("vehicle height", float, "metres"),

    # Road / route context
    "road_heading": ("nearest road node heading", float, "degrees"),
    "road_dist": ("distance to road centre", float, "metres"),
    "lane_offset": ("signed lateral offset from lane centre, + = right", float, "metres"),
    "lane_heading_delta": ("vehicle heading minus road heading", float, "degrees, -180..180"),
    "road_lanes": ("number of lanes at current road node", int, "0 if unknown"),
    "road_node2_heading": ("heading at next road node ahead", float, "degrees"),
    "road_node2_dist": ("distance to next road node", float, "metres"),
    "road_curve": ("heading change between nearest two road nodes", float, "degrees, -180..180"),
    "wp_x": ("active waypoint world X", float, "0 if no waypoint"),
    "wp_y": ("active waypoint world Y", float, "0 if no waypoint"),
    "wp_dist": ("distance to waypoint", float, "metres, 0 if no waypoint"),

    # Nearby mixed world state
    "near_total": ("total nearby entities gathered before mixed top-K", int, "0..N"),
    "near_entities_kept": ("number of entities kept in the mixed near_entities list", int, "0..MAX_NEARBY_TOTAL"),
    "near_vehs": ("all nearby vehicles within the gather radius", list, "variable length"),
    "near_peds": ("all nearby peds within the gather radius", list, "variable length"),
    "near_objects": ("all nearby driving-relevant static objects within the gather radius", list, "variable length"),
    "near_entities": ("mixed nearby entities ranked by distance and capped for tensor use", list, "sorted by distance"),
}

SCALAR_FIELDS = [k for k, v in FIELDS.items() if v[1] in (int, float, bool)]

WEATHER = {
    0: "EXTRASUNNY",
    1: "CLEAR",
    2: "NEUTRAL",
    3: "SMOG",
    4: "FOGGY",
    5: "OVERCAST",
    6: "CLOUDS",
    7: "CLEARING",
    8: "RAIN",
    9: "THUNDER",
    10: "LIGHTRAIN",
    11: "SNOWLIGHT",
    12: "BLIZZARD",
    13: "SNOW",
    14: "XMAS",
    15: "HALLOWEEN",
}

ENTITY_TYPES = {
    "vehicle": 1,
    "ped": 2,
    "object": 3,
}


@dataclass
class GameState:
    raw: Dict[str, Any]
    fields: List[str]

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {k: self.raw[k] for k in self.fields if k in self.raw}

    def to_list(self) -> List[float]:
        return [_as_float(self.raw.get(k, 0.0)) for k in self.fields]

    def to_tensor(self):
        if not _TORCH:
            raise ImportError("PyTorch not installed — use to_list() instead")
        import torch

        return torch.tensor(self.to_list(), dtype=torch.float32)

    def summary(self) -> str:
        r = self.raw
        lines = [
            f"[GTA5 State @ tick={r.get('ts', 0)}]",
            f"  Player pos=({r.get('px', 0):.1f}, {r.get('py', 0):.1f}, {r.get('pz', 0):.1f})"
            f" spd={(r.get('vx', 0) ** 2 + r.get('vy', 0) ** 2) ** 0.5:.1f} m/s",
            f"  HP={r.get('hp', 0):.0f}/{r.get('hp_max', 0):.0f} armor={r.get('armor', 0):.0f}"
            f" wanted={'*' * r.get('wanted', 0) or 'clean'}",
            f"  time={r.get('clock_h', 0):02d}:{r.get('clock_m', 0):02d}"
            f" weather={WEATHER.get(r.get('weather', 0), r.get('weather', 0))}",
        ]
        if "v_speed" in r:
            lines.append(
                f"  Vehicle spd={r.get('v_speed', 0) * 3.6:.1f} km/h"
                f" gear={r.get('v_gear', 0)}"
                f" steer={r.get('v_steer', 0):.2f}"
            )
        if "u_throttle" in r or "agent_active" in r:
            lines.append(
                f"  Input user=(thr={r.get('u_throttle', 0):.2f},"
                f" brk={r.get('u_brake', 0):.2f},"
                f" str={r.get('u_steer', 0):.2f},"
                f" hb={int(bool(r.get('u_handbrake', False)))})"
                f" agent={'on' if r.get('agent_active', False) else 'off'}"
                f" cmd=(thr={r.get('agent_throttle', 0):.2f},"
                f" brk={r.get('agent_brake', 0):.2f},"
                f" str={r.get('agent_steer', 0):.2f})"
            )
        lines.append(
            f"  Neighbors total={r.get('near_total', 0)}"
            f" mixed={r.get('near_entities_kept', len(r.get('near_entities', [])))}"
            f" vehs={len(r.get('near_vehs', []))}"
            f" peds={len(r.get('near_peds', []))}"
            f" objs={len(r.get('near_objects', []))}"
        )
        return "\n".join(lines)


# Legacy fixed layouts retained for compatibility with existing model code.
NEAR_VEH_FIELDS = [
    "rel_fwd",
    "rel_lat",
    "rel_z",
    "dist",
    "speed",
    "rel_v_fwd",
    "rel_v_lat",
    "heading",
    "hdiff",
    "dim_w",
    "dim_l",
    "dim_h",
    "eng_hp",
    "body_hp",
    "stopped",
    "has_driver",
    "ttc",
]

NEAR_PED_FIELDS = [
    "rel_fwd",
    "rel_lat",
    "rel_z",
    "dist",
    "speed",
    "rel_v_fwd",
    "rel_v_lat",
    "hdiff",
    "hp",
    "in_veh",
    "running",
    "crossing",
    "ttc",
]

NEAR_OBJECT_FIELDS = [
    "rel_fwd",
    "rel_lat",
    "rel_z",
    "dist",
    "dist3d",
    "speed",
    "rel_v_fwd",
    "rel_v_lat",
    "heading",
    "hdiff",
    "dim_w",
    "dim_l",
    "dim_h",
    "bucket_id",
    "has_collision",
    "is_visible",
    "is_on_roadside",
    "attached",
]

NEAR_ENTITY_FIELDS = [
    "type_id",
    "bucket_id",
    "dist",
    "dist3d",
    "rel_fwd",
    "rel_lat",
    "rel_z",
    "speed",
    "rel_v_fwd",
    "rel_v_lat",
    "heading",
    "hdiff",
    "dim_w",
    "dim_l",
    "dim_h",
    "is_static",
    "has_collision",
    "is_visible",
    "is_on_roadside",
    "ttc",
    "eng_hp",
    "body_hp",
    "hp",
    "class",
    "occupants",
    "has_driver",
    "in_veh",
    "running",
    "walking",
    "crossing",
    "attached",
]

_VEH_ZERO = [0.0] * len(NEAR_VEH_FIELDS)
_PED_ZERO = [0.0] * len(NEAR_PED_FIELDS)
_OBJ_ZERO = [0.0] * len(NEAR_OBJECT_FIELDS)
_ENTITY_ZERO = [0.0] * len(NEAR_ENTITY_FIELDS)


def _as_float(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    if value is None:
        return 0.0
    return float(value)


def _flatten_rows(rows: Iterable[Dict[str, Any]], field_order: List[str]) -> List[float]:
    out: List[float] = []
    for row in rows:
        for key in field_order:
            out.append(_as_float(row.get(key, 0.0)))
    return out


def flatten_nearby(
    state: GameState,
    max_vehs: int = 10,
    max_peds: int = 10,
) -> List[float]:
    """
    Legacy helper for older models that only consume nearby vehicles and peds.
    It ignores `near_objects` and the mixed `near_entities` list.
    """
    out: List[float] = []

    vehs = state.raw.get("near_vehs", [])[:max_vehs]
    out.extend(_flatten_rows(vehs, NEAR_VEH_FIELDS))
    for _ in range(max_vehs - len(vehs)):
        out.extend(_VEH_ZERO)

    peds = state.raw.get("near_peds", [])[:max_peds]
    out.extend(_flatten_rows(peds, NEAR_PED_FIELDS))
    for _ in range(max_peds - len(peds)):
        out.extend(_PED_ZERO)

    return out


def flatten_nearby_tensor(state: GameState, max_vehs: int = 10, max_peds: int = 10):
    if not _TORCH:
        raise ImportError("PyTorch not installed")
    import torch

    return torch.tensor(flatten_nearby(state, max_vehs, max_peds), dtype=torch.float32)


def flatten_objects(state: GameState, max_objects: int = 10) -> List[float]:
    objs = state.raw.get("near_objects", [])[:max_objects]
    out = _flatten_rows(objs, NEAR_OBJECT_FIELDS)
    for _ in range(max_objects - len(objs)):
        out.extend(_OBJ_ZERO)
    return out


def flatten_objects_tensor(state: GameState, max_objects: int = 10):
    if not _TORCH:
        raise ImportError("PyTorch not installed")
    import torch

    return torch.tensor(flatten_objects(state, max_objects), dtype=torch.float32)


def _fallback_world_entities(state: GameState) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for key, type_name in (
        ("near_vehs", "vehicle"),
        ("near_peds", "ped"),
        ("near_objects", "object"),
    ):
        for entry in state.raw.get(key, []):
            item = dict(entry)
            item.setdefault("entity_type", type_name)
            item.setdefault("type_id", ENTITY_TYPES[type_name])
            merged.append(item)
    merged.sort(key=lambda item: (item.get("rank", 10**9), item.get("dist", 10**9)))
    return merged


def flatten_world(state: GameState, max_entities: int = 32) -> List[float]:
    """
    Preferred helper for the expanded schema.

    Layout:
        [entity_0_fields ... entity_1_fields ...]

    Source order is `near_entities` when available, otherwise a rank/dist merge
    of the split per-type lists.
    """
    entities = state.raw.get("near_entities") or _fallback_world_entities(state)
    entities = entities[:max_entities]

    out = _flatten_rows(entities, NEAR_ENTITY_FIELDS)
    for _ in range(max_entities - len(entities)):
        out.extend(_ENTITY_ZERO)
    return out


def flatten_world_tensor(state: GameState, max_entities: int = 32):
    if not _TORCH:
        raise ImportError("PyTorch not installed")
    import torch

    return torch.tensor(flatten_world(state, max_entities), dtype=torch.float32)


class GameStateReader:
    """
    Connects to the GTA5State named pipe and pushes GameState objects
    into an internal queue.

    Parameters
    ----------
    fields : list[str] | None
        Which scalar fields to include in GameState.to_tensor().
        None means all SCALAR_FIELDS.
    maxsize : int
        Max queue depth — oldest frames are dropped when full.
    pipe_name : str
        Override the pipe path (rarely needed).
    """

    PIPE_PATH = r"\\.\pipe\GTA5State"

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        maxsize: int = 30,
        pipe_name: str = PIPE_PATH,
        log_dir: str = "logs",
    ):
        self.fields = fields if fields is not None else SCALAR_FIELDS
        self._q = queue.Queue(maxsize=maxsize)
        self._pipe = pipe_name
        self._stop = threading.Event()
        self._thread = None
        self._stats = {"frames": 0, "drops": 0, "errors": 0}
        self._log = _build_logger(log_dir)
        self._stat_interval = 30.0
        self._last_stat_t = 0.0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log.info(
            f"GameStateReader started | pipe={self._pipe} | "
            f"fields={len(self.fields)} | maxsize={self._q.maxsize}"
        )
        return self

    def stop(self):
        s = self._stats
        self._log.info(
            f"GameStateReader stopping | frames={s['frames']} "
            f"drops={s['drops']} errors={s['errors']}"
        )
        self._stop.set()

    def get(self, timeout: float = 5.0) -> GameState:
        return self._q.get(timeout=timeout)

    def get_nowait(self) -> Optional[GameState]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def _run(self):
        attempt = 0
        while not self._stop.is_set():
            try:
                attempt += 1
                self._connect_and_read()
            except Exception as exc:
                self._stats["errors"] += 1
                msg = f"Pipe error (attempt {attempt}): {exc} — reconnecting in 2s"
                if attempt == 1 or attempt % 30 == 0:
                    self._log.info(msg)
                else:
                    self._log.debug(msg)
                time.sleep(2.0)

    def _connect_and_read(self):
        import ctypes
        import ctypes.wintypes as wt

        GENERIC_READ = 0x80000000
        OPEN_EXISTING = 3
        ERROR_BROKEN_PIPE = 109
        ERROR_PIPE_NOT_CONNECTED = 233
        INVALID_HANDLE = wt.HANDLE(-1).value

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wt.HANDLE
        k32.ReadFile.restype = wt.BOOL
        k32.CloseHandle.restype = wt.BOOL

        self._log.debug(f"Connecting to {self._pipe} ...")
        handle = k32.CreateFileW(
            self._pipe, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None
        )

        if handle == INVALID_HANDLE or handle is None:
            err = ctypes.get_last_error()
            lvl = logging.DEBUG if err == 2 else logging.WARNING
            self._log.log(lvl, f"CreateFile failed — WinError {err}")
            raise OSError(f"CreateFile failed (err={err})")

        self._log.info("Pipe connected — receiving game state stream")
        connect_time = time.time()
        remainder = b""

        try:
            while not self._stop.is_set():
                chunk = ctypes.create_string_buffer(16384)
                n_read = wt.DWORD(0)
                ok = k32.ReadFile(handle, chunk, 16384, ctypes.byref(n_read), None)

                if not ok:
                    err = ctypes.get_last_error()
                    if err in (ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED):
                        self._log.info("Pipe closed by game")
                    else:
                        self._log.warning(f"ReadFile error WinError {err}")
                    break

                if n_read.value == 0:
                    continue

                remainder += chunk.raw[: n_read.value]

                while b"\n" in remainder:
                    line_b, remainder = remainder.split(b"\n", 1)
                    line_b = line_b.strip()
                    if not line_b:
                        continue
                    try:
                        data = json.loads(line_b.decode("utf-8"))
                        state = GameState(raw=data, fields=self.fields)
                        self._stats["frames"] += 1

                        if self._q.full():
                            try:
                                self._q.get_nowait()
                                self._stats["drops"] += 1
                            except queue.Empty:
                                pass
                        self._q.put(state)

                        now_t = time.time()
                        if now_t - self._last_stat_t >= self._stat_interval:
                            self._last_stat_t = now_t
                            s = self._stats
                            elapsed = now_t - connect_time
                            hz = s["frames"] / elapsed if elapsed > 0 else 0.0
                            self._log.info(
                                f"STATS | frames={s['frames']} drops={s['drops']} "
                                f"errors={s['errors']} rate={hz:.1f} Hz "
                                f"queue={self._q.qsize()}/{self._q.maxsize}"
                            )

                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        self._stats["errors"] += 1
                        self._log.warning(f"Decode error: {exc} | raw={line_b[:80]!r}")

        finally:
            k32.CloseHandle(handle)
            duration = time.time() - connect_time
            s = self._stats
            self._log.info(
                f"Pipe disconnected after {duration:.1f}s | "
                f"frames={s['frames']} drops={s['drops']} errors={s['errors']}"
            )


if __name__ == "__main__":
    import argparse
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_stats_dir = os.path.join(script_dir, "stats")

    parser = argparse.ArgumentParser(description="GTA5 State Reader")
    parser.add_argument("--save", action="store_true", help="Save every frame as JSON to the stats folder")
    parser.add_argument(
        "--stats-dir",
        default=default_stats_dir,
        help="Output directory for saved frames (default: gta_stream/stats)",
    )
    args = parser.parse_args()

    stats_dir = args.stats_dir
    if args.save:
        os.makedirs(stats_dir, exist_ok=True)
        next_frame_num = _find_next_saved_frame_number(stats_dir)
        print(f"Saving frames to {os.path.abspath(stats_dir)}/")
        print(f"Starting at frame_{next_frame_num:06d}.json")

    reader = GameStateReader(fields=SCALAR_FIELDS).start()

    last_print = 0.0
    print_every = 5.0
    if args.save:
        frame_num = next_frame_num - 1
    else:
        frame_num = 0

    try:
        while True:
            try:
                state = reader.get(timeout=5.0)
            except queue.Empty:
                continue

            frame_num += 1

            if args.save:
                filepath = os.path.join(stats_dir, f"frame_{frame_num:06d}.json")
                with open(filepath, "w", encoding="utf-8") as fh:
                    json.dump(state.raw, fh, indent=2)

            now_t = time.time()
            if now_t - last_print < print_every:
                continue
            last_print = now_t

            s = reader.stats
            print(state.summary())
            saved_msg = f"  saved={frame_num}" if args.save else ""
            print(f"  frames={s['frames']}  drops={s['drops']}  errors={s['errors']}{saved_msg}\n")

    except KeyboardInterrupt:
        reader.stop()
        sys.exit(0)
