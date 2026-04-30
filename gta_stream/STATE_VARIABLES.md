# GTA5 State Variables

Stream is vehicle-only: frames are emitted only while the player is inside a
vehicle.

Default rate: about 20 Hz over `\\.\pipe\GTA5State`.

Inbound drive commands are accepted over `\\.\pipe\GTA5Control` as line-delimited
JSON objects with `enabled`, `steer`, `throttle`, `brake`, and `handbrake`
fields.

---

## Top-Level Scalar Fields

### Player

| Key | Type | Notes |
|-----|------|-------|
| `ts` | int | monotonic game tick timestamp |
| `px`, `py`, `pz` | float | player world position |
| `vx`, `vy`, `vz` | float | player world velocity |
| `rx`, `ry`, `rz` | float | player rotation |
| `hp`, `hp_max` | float | health values |
| `armor` | float | armour |
| `wanted` | int | wanted level |
| `dead` | bool | player dead flag |

### Environment

| Key | Type | Notes |
|-----|------|-------|
| `clock_h`, `clock_m`, `clock_s` | int | in-game clock |
| `weather` | int | weather enum |
| `rain` | float | rainfall intensity |
| `wind_speed` | float | wind speed |
| `wind_x`, `wind_y` | float | wind direction vector |

### Ego Vehicle

| Key | Type | Notes |
|-----|------|-------|
| `v_speed` | float | vehicle speed, m/s |
| `v_throttle`, `v_brake`, `v_steer` | float | current controls |
| `u_throttle`, `u_brake`, `u_steer` | float | raw player inputs before automation override |
| `u_handbrake` | bool | raw player handbrake input |
| `u_horn` | bool | raw player horn input |
| `u_look_behind` | bool | raw player look-behind input |
| `u_duck` | bool | raw player duck input |
| `agent_active` | bool | automation currently being applied |
| `agent_throttle`, `agent_brake`, `agent_steer` | float | latest automation command values |
| `agent_handbrake` | bool | latest automation handbrake command |
| `agent_age_ms` | int | age of the latest automation command |
| `v_rpm` | float | engine RPM (0.0-1.0 normalized) |
| `v_gear`, `v_max_gear` | int | current gear and highest gear |
| `v_steer_angle` | float | actual wheel steering angle in degrees |
| `v_clutch` | float | clutch engagement (0.0-1.0) |
| `v_wheel_speed` | float | drive wheel rotational speed (m/s) |
| `v_accel` | float | current acceleration |
| `v_engine_hp`, `v_body_hp` | float | engine/body health |
| `v_engine_on` | bool | engine running |
| `v_turbo`, `v_fuel` | float | currently placeholders in this script |
| `v_model` | int | model hash |
| `v_class` | int | GTA vehicle class |
| `v_ax`, `v_ay`, `v_az` | float | local speed-vector components |
| `v_heading` | float | vehicle heading |
| `v_fwd_x`, `v_fwd_y` | float | forward vector |
| `v_right_x`, `v_right_y` | float | right vector |
| `v_dim_w`, `v_dim_l`, `v_dim_h` | float | vehicle dimensions |

### Road / Route Context

| Key | Type | Notes |
|-----|------|-------|
| `road_heading` | float | nearest road node heading |
| `road_dist` | float | distance to road centre |
| `lane_offset` | float | signed lateral offset from lane centre (+ = right of centre) |
| `lane_heading_delta` | float | vehicle heading minus road heading, -180..180 |
| `road_lanes` | int | number of lanes at current road node |
| `road_node2_heading` | float | heading at next road node ahead |
| `road_node2_dist` | float | distance to next road node |
| `road_curve` | float | heading change between nearest two road nodes, -180..180 |
| `wp_x`, `wp_y` | float | active waypoint coordinates |
| `wp_dist` | float | distance to waypoint |

---

## Nearby World State

Each frame also includes:

| Key | Type | Notes |
|-----|------|-------|
| `near_total` | int | total nearby entities gathered before mixed top-K |
| `near_entities_kept` | int | number of entities retained in `near_entities` |
| `near_vehs` | list[dict] | all nearby vehicles within the gather radius |
| `near_peds` | list[dict] | all nearby pedestrians within the gather radius |
| `near_objects` | list[dict] | all nearby driving-relevant static objects within the gather radius |
| `near_entities` | list[dict] | mixed nearby entities sorted by distance and capped for tensor use |

The mixed pool is globally ranked by distance across all types, then truncated
to the configured `MAX_NEARBY_TOTAL`. The split per-type lists are not truncated
by other types.

---

## Common Nearby Entity Fields

Every entry in `near_entities` contains the following shared fields. Split lists
(`near_vehs`, `near_peds`, `near_objects`) contain the same fields plus
type-specific extras.

| Key | Type | Meaning |
|-----|------|---------|
| `rank` | int | global rank in the mixed nearby list |
| `type_id` | int | `1=vehicle`, `2=ped`, `3=object` |
| `entity_type` | str | `vehicle`, `ped`, or `object` |
| `bucket_id` | int | coarse semantic class id |
| `semantic_bucket` | str | coarse semantic class name |
| `model_hash` | int | model hash |
| `hash` | int | alias of `model_hash` for compatibility |
| `x`, `y`, `z` | float | world position |
| `rel_fwd` | float | ego-forward displacement, positive means ahead |
| `rel_lat` | float | ego-right displacement, positive means right |
| `rel_z` | float | vertical offset relative to ego |
| `dist` | float | 2D distance from ego |
| `dist3d` | float | full 3D distance from ego |
| `speed` | float | absolute speed |
| `vx`, `vy`, `vz` | float | world velocity |
| `rel_v_fwd` | float | relative velocity along ego forward axis |
| `rel_v_lat` | float | relative velocity along ego lateral axis |
| `heading` | float | world heading |
| `hdiff` | float | heading difference from ego, `-180..180` |
| `fwd_x`, `fwd_y` | float | entity forward vector |
| `dim_w`, `dim_l`, `dim_h` | float | bounding-box dimensions |
| `is_static` | bool | true for exported static objects |
| `has_collision` | bool | collision enabled |
| `is_visible` | bool | entity visible flag |
| `is_on_roadside` | bool | derived near-roadside band flag |

---

## Vehicle-Specific Fields

Entries in `near_vehs` also include:

| Key | Type | Meaning |
|-----|------|---------|
| `class` | int | GTA vehicle class |
| `eng_hp` | float | engine health |
| `body_hp` | float | body health |
| `stopped` | bool | stationary vehicle flag |
| `occupants` | int | passenger count excluding driver |
| `has_driver` | bool | driver present |
| `ttc` | float | simple linear time-to-collision estimate |

Vehicle buckets currently include `vehicle`, `motorcycle`, `bicycle`,
`train`, `emergency_vehicle`, and `service_vehicle`.

---

## Ped-Specific Fields

Entries in `near_peds` also include:

| Key | Type | Meaning |
|-----|------|---------|
| `hp` | float | ped health |
| `in_veh` | bool | ped inside a vehicle |
| `running` | bool | ped running |
| `walking` | bool | ped walking |
| `crossing` | bool | ped likely crossing ego path |
| `ttc` | float | simple linear time-to-collision estimate |

Ped bucket is currently `pedestrian`.

---

## Object-Specific Fields

Entries in `near_objects` also include:

| Key | Type | Meaning |
|-----|------|---------|
| `attached` | bool | object attached to another entity |
| `ttc` | float | currently `0` for objects |

Current object semantic buckets:

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

Bucket assignment uses a small curated model-hash table first, then falls back
to dimension and collision heuristics.

---

## Tensor Helpers

The new `u_*` and `agent_*` fields are scalar fields only. They are not packed
into `flatten_world()` because they describe ego control state rather than
nearby entities.

### Preferred mixed-world helper

```python
from game_state_reader import flatten_world, flatten_world_tensor

state = reader.get()
world_vec = flatten_world(state, max_entities=32)
world_tensor = flatten_world_tensor(state, max_entities=32)
```

Each slot uses the following field order:

```text
type_id, bucket_id, dist, dist3d, rel_fwd, rel_lat, rel_z,
speed, rel_v_fwd, rel_v_lat, heading, hdiff,
dim_w, dim_l, dim_h, is_static, has_collision, is_visible, is_on_roadside,
ttc, eng_hp, body_hp, hp, class, occupants, has_driver,
in_veh, running, walking, crossing, attached
```

Missing slots are zero-padded.

### Legacy vehicle/ped helper

```python
from game_state_reader import flatten_nearby, flatten_nearby_tensor
```

This helper keeps the older vehicle-plus-ped layout and ignores objects.

### Optional object-only helper

```python
from game_state_reader import flatten_objects, flatten_objects_tensor
```

This helper emits only `near_objects` with a fixed number of slots.
