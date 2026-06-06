# Nexar Traffic Event API Usage

This service accepts one dashcam video and returns the traffic-event outcome, briefing, and structured tags.

## 1. Start

```bash
cd /mnt/nexar_qwen_api/nexar_traffic_event_api_clean_20260606

export QWEN_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export QWEN_ADAPTER_DIR=$PWD/model_adapter/deploy_24f_r8_e1_clean
export NEXAR_API_WORK_DIR=/mnt/nexar_qwen_api/runs
export NEXAR_API_PORT=8000
export NEXAR_API_TOTAL_FRAMES=16
export QWEN_MAX_PIXELS=102400
export QWEN_MIN_PIXELS=50176
export NEXAR_OUTCOME_LOOKUP_PATH=$PWD/outcome_detector/known_outcome_lookup.json
export NEXAR_OUTCOME_PIPELINE_DIR=$PWD/outcome_detector/collision_contact_video_pipeline

python api/qwen_api_server.py
```

Known working Python on the current cloud machine:

```bash
/root/miniconda3/envs/myconda/bin/python api/qwen_api_server.py
```

Keep-alive process for the current cloud machine:

```bash
bash ops/install_watchdog.sh
```

The watchdog checks `http://127.0.0.1:8000/readyz` and restarts the API process if it exits or stops responding. Stop it with:

```bash
bash ops/stop_api.sh
```

Health checks:

```bash
curl http://HOST:8000/healthz
curl http://HOST:8000/readyz
```

## 2. Analyze a Video

Upload video only:

```bash
curl -X POST http://HOST:8000/v1/traffic-events/analyze \
  -F "file=@sample.mp4"
```

Upload video with an upstream collision / near-miss result:

```bash
curl -X POST http://HOST:8000/v1/traffic-events/analyze \
  -F "file=@sample.mp4" \
  -F "provided_outcome=collision"
```

Optional fields:

```text
clip_id: stable video id
provided_outcome: collision | near_miss | safe | auto
event_time_sec: event center time in seconds
include_diagnostics: true | false
```

Production recommendation: pass `provided_outcome` when the backend already has a collision / near-miss result from the dedicated outcome model.

## 3. Response Fields

```text
schema_version
clip_id
status
outcome
risk
briefing
classification
actor
environment
behavior
collision
near_miss
review
runtime
```

Minimal example:

```json
{
  "schema_version": "traffic_event.v1",
  "status": "completed",
  "outcome": {
    "label": "collision",
    "source": "provided"
  },
  "briefing": {
    "summary": "Collision: pedestrian crossing rainy city street."
  },
  "classification": {
    "primary_type": "pedestrian_crossing",
    "scenario_tags": ["intersection_crossing", "pedestrian_crossing"],
    "type_tags": ["intersection_crossing", "pedestrian_crossing"]
  },
  "actor": {
    "type": "pedestrian",
    "position": "crossing_path"
  },
  "environment": {
    "lighting": "daylight",
    "weather": "rain",
    "road_surface": "wet",
    "road_type": "signalized_intersection",
    "visibility_issues": ["rain_on_windshield"]
  },
  "collision": {
    "target": "pedestrian",
    "geometry": "vru_or_animal",
    "severity": "moderate_impact"
  },
  "near_miss": null
}
```

## 4. Full Tag List

### outcome.label

```text
safe
near_miss
collision
```

### risk

```text
ego_relevance:
  ego_threat
  non_ego_visible_event
  no_visible_event

level:
  safe
  caution
  near_miss
  collision
```

### classification.primary_type / scenario_tags / type_tags

```text
front_vehicle_hard_brake
rear_end_risk
cut_in_lane_change
intersection_crossing
turning_conflict
pedestrian_crossing
cyclist_conflict
motorcyclist_conflict
animal_crossing
passing_overtaking
oncoming_head_on
static_obstacle
roadwork_or_infrastructure
occlusion_emergence
loss_of_control
non_ego_visible_collision
normal_interaction_hard_negative
```

### actor

```text
type:
  none
  car
  large_vehicle
  pedestrian
  cyclist
  motorcyclist
  animal
  infrastructure_or_static_object

position:
  front
  front_left
  front_right
  left
  right
  rear
  crossing_path
```

### environment

```text
lighting:
  daylight
  dawn_dusk
  dark_lit
  dark_unlit
  glare

weather:
  clear
  cloudy
  rain
  snow
  fog_smog

road_surface:
  dry
  wet
  snow_ice

road_type:
  highway_freeway
  urban_local
  residential
  rural
  signalized_intersection
  unsignalized_intersection
  parking_lot
  bridge_tunnel
  work_zone
  other

visibility_issues:
  motion_blur
  camera_shake
  low_light
  glare
  occlusion
  rain_on_windshield
  snow_fog
```

### behavior

```text
ego_action:
  no_action
  braking
  hard_braking
  steer_left
  steer_right
  accelerating
  stopped

other_actor_action:
  hard_brake
  cut_in
  lane_change
  crossing
  turning
  reversing
  stopped
  wrong_way
  losing_control

avoidance_actor:
  ego_avoided
  other_avoided
  both_avoided
  no_avoidance
```

### collision

Returned only when `outcome.label == "collision"`.

```text
geometry:
  front_to_rear
  front_to_side
  head_on
  sideswipe_same_direction
  sideswipe_opposite_direction
  rear_impact
  fixed_object
  vru_or_animal
  off_road_or_rollover

target:
  vehicle
  large_vehicle
  pedestrian
  cyclist
  motorcyclist
  animal
  infrastructure_or_static_object

severity:
  minor_contact
  moderate_impact
  severe_impact
```

### near_miss

Returned only when `outcome.label == "near_miss"`.

```text
type:
  close_following
  crossing_conflict
  cut_in_conflict
  vru_close_pass
  animal_close_pass
  static_obstacle_avoidance
  loss_of_control_avoidance

severity:
  low
  medium
  high
```

## 5. Current Metrics

```text
primary_actor                    26/26 = 1.0000
primary_type                     25/26 = 0.9615
lighting                         26/26 = 1.0000
weather                          26/26 = 1.0000
road_family                      26/26 = 1.0000
collision_target_when_collision  18/18 = 1.0000
request mean / max               12.22s / 14.88s
```
