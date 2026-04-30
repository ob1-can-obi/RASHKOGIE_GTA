# GTA5 State Streamer

Streams live GTA V driving state from a SHVDN C# script into Python over a
Windows named pipe.

The exporter now includes:
- Ego player and vehicle state
- Raw player driving inputs (`u_throttle`, `u_brake`, `u_steer`, `u_handbrake`)
- Agent command echo fields (`agent_*`) so you can see what automation is sending
- Full nearby vehicle lists
- Full nearby pedestrian lists
- Full nearby driving-relevant static object lists such as traffic lights, poles, signs,
  barriers, walls, trees, bushes, vegetation, and other roadside props
- A mixed `near_entities` list ranked globally by distance and capped for direct model use
- A second named pipe for inbound drive commands

The stream is still vehicle-only: frames are emitted only while the player is
inside a vehicle.

---

## Files

| File | Purpose |
|------|---------|
| `GTA5_StateStreamer.cs` | SHVDN script running inside GTA V |
| `game_state_reader.py` | Python reader and tensor helpers |
| `game_control.py` | Python named-pipe client for steering/throttle/brake commands |
| `action_adapter.py` | Converts planner tensors into GTA command timelines |
| `run_action_planner.py` | Loads a trained planner checkpoint and drives GTA through the control pipe |
| `gta_ws_bridge.py` | Full-duplex WebSocket bridge for live state + control |
| `requirements.txt` | Python dependency for the WebSocket bridge |
| `run_ws_bridge.bat` | Windows launcher for the WebSocket bridge |
| `STATE_VARIABLES.md` | Field reference and tensor layout |

---

## Data Flow

```text
GTA V Enhanced
  -> GTA5_StateStreamer.cs
  -> \\.\pipe\GTA5State
  -> game_state_reader.py
  -> your training / imitation-learning loop

your policy / controller
  -> game_control.py
  -> \\.\pipe\GTA5Control
  -> GTA5_StateStreamer.cs
  -> in-game vehicle controls

optional network bridge
  -> gta_ws_bridge.py
  -> ws://127.0.0.1:8765
  -> browser / Python / remote controller
```

The C# side writes one JSON object per line at roughly 20 Hz by default. The
Python state reader reconnects automatically if the game or script restarts.
The control pipe expects line-delimited JSON commands at roughly 20 Hz or
faster. A short watchdog returns control to the player if commands stop.

If you want a single seamless socket interface instead of opening named pipes
from your controller, run `gta_ws_bridge.py`. It exposes both live game state
and drive commands over one WebSocket connection.

---

## Quick Start

1. Install `ScriptHookV.dll` and `dinput8.dll` into your GTA V root.
2. Install `ScriptHookVDotNet3.dll` into the same root.
3. Copy `GTA5_StateStreamer.cs` into `D:\Steam\GTAVEnhanced\scripts\`.
4. Install Python deps:

```bash
pip install torch websockets
```

5. Start GTA V, load into a drivable session, then run the Python tools from
   Windows Python so they can open Windows named pipes:

```bash
cd gta_stream
python3 game_state_reader.py
```

To save raw frames while you drive:

```bash
python3 game_state_reader.py --save
```

To send a single drive command:

```bash
python3 game_control.py --steer 0.10 --throttle 0.35
```

To release automation immediately:

```bash
python3 game_control.py --disable
```

To expose state + control over a WebSocket server:

```bash
python3 gta_ws_bridge.py --host 127.0.0.1 --port 8765
```

Or on Windows:

```bat
run_ws_bridge.bat
```

---

## Using It From Python

### Ego-only scalar input

```python
from game_state_reader import GameStateReader

EGO_FIELDS = [
    "v_speed",
    "v_steer",
    "v_throttle",
    "v_brake",
    "v_heading",
    "road_heading",
    "road_dist",
    "lane_offset",
    "lane_heading_delta",
    "road_lanes",
    "road_curve",
    "wp_dist",
    "rain",
]

reader = GameStateReader(fields=EGO_FIELDS).start()
state = reader.get()
ego = state.to_tensor()
```

### Raw player input capture

```python
from game_state_reader import GameStateReader

reader = GameStateReader().start()

while True:
    state = reader.get()
    player_action = {
        "steer": state["u_steer"],
        "throttle": state["u_throttle"],
        "brake": state["u_brake"],
        "handbrake": state["u_handbrake"],
    }
```

### Preferred mixed-world tensor

```python
import torch
from game_state_reader import GameStateReader, flatten_world_tensor

reader = GameStateReader().start()

while True:
    state = reader.get()
    ego = state.to_tensor()
    world = flatten_world_tensor(state, max_entities=32)
    x = torch.cat([ego, world])
```

`flatten_world_tensor()` uses the mixed `near_entities` list, which is globally
ranked by distance across vehicles, peds, and nearby objects.

### Legacy vehicle/ped tensor

```python
from game_state_reader import flatten_nearby_tensor

legacy = flatten_nearby_tensor(state, max_vehs=10, max_peds=10)
```

This helper is kept for backward compatibility. It ignores static objects and
does not use the mixed global ranking.

### Sending programmatic driving commands

```python
from game_control import GameControlClient
from game_state_reader import GameStateReader

reader = GameStateReader().start()

with GameControlClient() as control:
    while True:
        state = reader.get()

        steer_error = (state["road_heading"] - state["v_heading"]) / 45.0
        steer = max(-1.0, min(1.0, steer_error))

        target_speed = 12.0  # m/s
        throttle = 0.45 if state["v_speed"] < target_speed else 0.0
        brake = 0.2 if state["v_speed"] > target_speed + 2.0 else 0.0

        control.send(steer=steer, throttle=throttle, brake=brake)
```

The control command schema is:

```json
{"enabled": true, "steer": 0.0, "throttle": 0.0, "brake": 0.0, "handbrake": false}
```

Send commands continuously. The game-side watchdog drops automation if the
control client stops sending updates for about 250 ms.

### Using the WebSocket bridge

Run the bridge on the Windows machine where GTA is running:

```bash
python3 gta_ws_bridge.py --host 127.0.0.1 --port 8765
```

Then connect from any WebSocket client to:

```text
ws://127.0.0.1:8765
```

Server messages:

```json
{"type":"hello","client_id":"8f4a6f5c","has_state":true}
{"type":"state","seq":42,"data":{"v_speed":11.8,"u_steer":-0.04,"agent_active":false}}
{"type":"ack","action":"control","sent_at_ms":1713380000000}
```

Client messages:

```json
{"type":"ping"}
{"type":"get_state"}
{"type":"neutral"}
{"type":"disable"}
{"type":"control","steer":0.10,"throttle":0.45,"brake":0.00,"handbrake":false}
```

Minimal browser client:

```html
<script>
const ws = new WebSocket("ws://127.0.0.1:8765");

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "state") {
    console.log("speed", msg.data.v_speed, "user steer", msg.data.u_steer);
  }
};

function sendControl(steer, throttle, brake) {
  ws.send(JSON.stringify({
    type: "control",
    steer,
    throttle,
    brake,
    handbrake: false
  }));
}
</script>
```

Minimal Python client:

```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({
            "type": "control",
            "steer": 0.05,
            "throttle": 0.4,
            "brake": 0.0,
        }))

        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "state":
                print(msg["data"]["v_speed"], msg["data"]["u_steer"])

asyncio.run(main())
```

---

## Output Structure

Each frame contains:
- Scalar ego fields such as position, health, vehicle speed, and waypoint distance
- Lane and road context: `road_heading`, `road_dist`, `lane_offset`, `lane_heading_delta`, `road_lanes`, `road_node2_heading`, `road_node2_dist`, `road_curve`
- Raw player input fields `u_throttle`, `u_brake`, `u_steer`, `u_handbrake`
- Agent command echo fields `agent_active`, `agent_throttle`, `agent_brake`, `agent_steer`, `agent_handbrake`, `agent_age_ms`
- `near_total`: total nearby entities gathered before the mixed cap
- `near_entities_kept`: number of nearby entities retained in `near_entities`
- `near_vehs`: all nearby vehicles gathered within the radius
- `near_peds`: all nearby pedestrians gathered within the radius
- `near_objects`: all nearby driving-relevant static objects gathered within the radius
- `near_entities`: mixed nearby entities sorted by distance and capped for tensor use

Each entry in `near_entities` includes shared geometry and motion fields such as:
- `type_id`, `entity_type`
- `bucket_id`, `semantic_bucket`
- `model_hash`
- `x`, `y`, `z`
- `rel_fwd`, `rel_lat`, `rel_z`
- `dist`, `dist3d`
- `speed`, `vx`, `vy`, `vz`
- `rel_v_fwd`, `rel_v_lat`
- `heading`, `hdiff`
- `dim_w`, `dim_l`, `dim_h`
- `is_static`, `has_collision`, `is_visible`, `is_on_roadside`

Type-specific extras are also included:
- Vehicles: `class`, `eng_hp`, `body_hp`, `occupants`, `has_driver`, `stopped`, `ttc`
- Peds: `hp`, `in_veh`, `running`, `walking`, `crossing`, `ttc`
- Objects: `attached`

---

## Semantic Buckets

Objects are bucketed into coarse driving-relevant classes. Current buckets
include:

- `traffic_light`
- `street_light`
- `pole`
- `sign`
- `barrier`
- `curb`
- `wall`
- `tree`
- `bush`
- `vegetation`
- `building_edge`
- `roadside_prop`
- `unknown`

Bucket assignment uses a small curated model-hash map first, then falls back to
dimension and collision heuristics. Unknown nearby objects are still exported.

---

## Tuning

Edit these constants in `GTA5_StateStreamer.cs`:

```csharp
private const int   TICK_INTERVAL_MS = 50;   // 20 Hz
private const int   MAX_NEARBY_TOTAL = 32;   // global top-K across all types
private const float NEARBY_RADIUS    = 120f; // metres
```

Queue depth is controlled from Python:

```python
reader = GameStateReader(maxsize=60)
```

---

## Notes

- The stream is silent while on foot.
- `v_throttle`, `v_brake`, and `v_steer` are the effective in-game controls.
  `u_*` fields are the raw player inputs sampled before automation overrides.
- `near_entities` is the preferred source for training because it preserves a
  stable global ranking across mixed entity types.
- `v_fuel` is still placeholder zero (no reliable native). All other drivetrain
  fields (`v_rpm`, `v_gear`, `v_steer_angle`, `v_clutch`, `v_wheel_speed`,
  `v_accel`, `v_turbo`) now read from SHVDN Vehicle properties.

---

## Logs

Game-side log:

```text
D:\Steam\GTAVEnhanced\scripts\GTA5StateStreamer.log
```

Python-side log:

```text
gta_stream/logs/game_state_reader.log
```

Both sides automatically rotate logs when files grow too large.
