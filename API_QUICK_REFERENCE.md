# OrchestraAI API - Quick Reference

## Base URL
```
http://localhost:8000
```

## All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/status` | Server status |
| POST | `/episode/start` | Start episode |
| GET | `/episode/{id}` | Get episode state |
| POST | `/episode/{id}/step` | Execute action |
| POST | `/episode/{id}/end` | End episode |
| GET | `/episodes` | List episodes |
| GET | `/episode/{id}/history` | Get episode history |
| GET | `/training/metrics` | Get metrics |
| GET | `/training/metrics/history` | Get metrics history |
| GET | `/model/info` | Get model info |
| POST | `/model/checkpoint` | Save checkpoint |
| WS | `/ws/live` | WebSocket updates |

---

## Quick Examples

### Start Episode
```bash
curl -X POST http://localhost:8000/episode/start \
  -H "Content-Type: application/json" \
  -d '{"config": {"num_workers": 4, "budget": 1000}}'
```

### Execute Action
```bash
curl -X POST http://localhost:8000/episode/ep_abc123/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 0, "target_worker_id": 1}'
```

### Get Metrics
```bash
curl http://localhost:8000/training/metrics
```

### List Episodes
```bash
curl http://localhost:8000/episodes
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Action IDs

| ID | Action | Cost |
|----|--------|------|
| 0 | Assign subtask | 10 |
| 1 | Check output | 50 |
| 2 | Correct worker | 30 |
| 3 | Reassign task | 40 |
| 4 | Fire & replace | 100 |
| 5 | Approve output | 5 |
| 6 | Request clarification | 20 |

---

## Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request |
| 404 | Not found |
| 500 | Server error |

---

## Key Fields

### Observation
- `task_embedding` - 64-dim task vector
- `worker_states` - 4x5 worker matrix
- `subtask_status` - 4-element binary array
- `budget_remaining` - 0.0 to 1.0
- `steps_remaining` - 0.0 to 1.0

### Metrics
- `total_timesteps` - Training steps
- `mean_reward` - Average reward
- `episode_count` - Episodes completed
- `hallucination_detection_rate` - Detection %
- `average_episode_length` - Avg steps

---

## Documentation

- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Full Docs**: See `ORCHESTRAAI_API_DOCS.md`

---

## Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Start episode
r = requests.post(f"{BASE_URL}/episode/start", 
  json={"config": {"num_workers": 4}})
ep = r.json()

# Step
r = requests.post(f"{BASE_URL}/episode/{ep['episode_id']}/step",
  json={"action_id": 0})

# Metrics
r = requests.get(f"{BASE_URL}/training/metrics")
print(r.json())
```

---

**Version**: 1.0.0  
**Status**: ✅ Ready
