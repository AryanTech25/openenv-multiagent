# OrchestraAI API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. In production, add API key authentication.

---

## Root Endpoints

### GET /
Get API information and available endpoints.

**Response (200)**
```json
{
  "name": "Manager-Worker RL Environment API",
  "version": "1.0.0",
  "description": "Manager-Worker RL Environment API",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Health & Status Endpoints

### GET /health
Health check endpoint. Returns server health status.

**Response (200)**
```json
{
  "status": "healthy",
  "uptime_seconds": 123.45,
  "active_episodes": 0,
  "total_episodes": 0,
  "version": "1.0.0",
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

**Status Values**
- `healthy` - Server is running normally
- `degraded` - Server is running but with issues
- `unhealthy` - Server has critical issues

---

### GET /status
Get detailed server status.

**Response (200)**
```json
{
  "version": "1.0.0",
  "status": "running",
  "active_connections": 5,
  "active_episodes": 2,
  "uptime_seconds": 456.78,
  "environment": {
    "python_version": "3.14.3",
    "framework": "fastapi",
    "database": "mongodb"
  },
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

---

## Episode Management Endpoints

### POST /episode/start
Start a new episode.

**Request Body**
```json
{
  "config": {
    "num_workers": 4,
    "budget": 1000,
    "max_steps": 500
  }
}
```

**Response (200)**
```json
{
  "episode_id": "ep_abc123def456",
  "observation": {
    "task_embedding": [-0.5, 0.3, ...],
    "worker_states": [[0.8, 0.2, ...], ...],
    "subtask_status": [0, 0, 1, 0],
    "budget_remaining": 1.0,
    "steps_remaining": 1.0
  },
  "created_at": "2026-04-25T12:00:00.000000"
}
```

**Error Responses**
- `400` - Invalid configuration
- `500` - Server error

---

### GET /episode/{episode_id}
Get current state of an episode.

**Path Parameters**
- `episode_id` (string, required) - Episode ID

**Response (200)**
```json
{
  "episode_id": "ep_abc123def456",
  "observation": {
    "task_embedding": [-0.5, 0.3, ...],
    "worker_states": [[0.8, 0.2, ...], ...],
    "subtask_status": [0, 0, 1, 0],
    "budget_remaining": 0.85,
    "steps_remaining": 0.9
  },
  "total_reward": 45.3,
  "step_count": 50,
  "is_active": true,
  "created_at": "2026-04-25T12:00:00.000000"
}
```

**Error Responses**
- `404` - Episode not found
- `500` - Server error

---

### POST /episode/{episode_id}/step
Execute an action in an episode.

**Path Parameters**
- `episode_id` (string, required) - Episode ID

**Request Body**
```json
{
  "action_id": 0,
  "target_worker_id": 1
}
```

**Action IDs**
- `0` - Assign subtask (10 tokens)
- `1` - Check worker output (50 tokens)
- `2` - Correct worker (30 tokens)
- `3` - Reassign task (40 tokens)
- `4` - Fire and replace (100 tokens)
- `5` - Approve output (5 tokens)
- `6` - Request clarification (20 tokens)

**Response (200)**
```json
{
  "observation": {
    "task_embedding": [-0.5, 0.3, ...],
    "worker_states": [[0.8, 0.2, ...], ...],
    "subtask_status": [0, 1, 1, 0],
    "budget_remaining": 0.80,
    "steps_remaining": 0.88
  },
  "reward": 5.2,
  "done": false,
  "info": {
    "action_cost": 10,
    "worker_quality": 0.85,
    "hallucination_detected": false
  },
  "step_number": 51
}
```

**Error Responses**
- `400` - Invalid action
- `404` - Episode not found
- `500` - Server error

---

### POST /episode/{episode_id}/end
End an episode.

**Path Parameters**
- `episode_id` (string, required) - Episode ID

**Response (200)**
```json
{
  "episode_id": "ep_abc123def456",
  "final_reward": 125.5,
  "final_quality": 0.92,
  "total_steps": 150,
  "episode_statistics": {
    "total_tokens_used": 850,
    "hallucinations_detected": 3,
    "workers_replaced": 1,
    "tasks_completed": 5,
    "average_quality": 0.88
  }
}
```

**Error Responses**
- `404` - Episode not found
- `500` - Server error

---

### GET /episodes
List all episodes.

**Query Parameters**
- `skip` (integer, optional) - Number of episodes to skip (default: 0)
- `limit` (integer, optional) - Maximum episodes to return (default: 10)

**Response (200)**
```json
{
  "episodes": [
    {
      "episode_id": "ep_abc123def456",
      "status": "active",
      "reward": 45.3,
      "steps": 50,
      "created_at": "2026-04-25T12:00:00.000000"
    },
    {
      "episode_id": "ep_xyz789uvw012",
      "status": "completed",
      "reward": 125.5,
      "steps": 150,
      "created_at": "2026-04-25T11:50:00.000000"
    }
  ],
  "total_count": 2,
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

---

### GET /episode/{episode_id}/history
Get episode history/log.

**Path Parameters**
- `episode_id` (string, required) - Episode ID

**Response (200)**
```json
{
  "episode_id": "ep_abc123def456",
  "steps": [
    {
      "step": 1,
      "action": "assign_subtask",
      "reward": 0.5,
      "observation": {...},
      "timestamp": "2026-04-25T12:00:01.000000"
    },
    {
      "step": 2,
      "action": "check_worker_output",
      "reward": 2.3,
      "observation": {...},
      "timestamp": "2026-04-25T12:00:02.000000"
    }
  ],
  "total_reward": 45.3,
  "final_quality": 0.85,
  "created_at": "2026-04-25T12:00:00.000000",
  "ended_at": "2026-04-25T12:05:00.000000"
}
```

---

## Metrics Endpoints

### GET /training/metrics
Get current training metrics.

**Response (200)**
```json
{
  "total_timesteps": 50000,
  "mean_reward": 52.3,
  "episode_count": 100,
  "learning_rate": 0.0003,
  "hallucination_detection_rate": 0.87,
  "average_episode_length": 150,
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

**Metrics Explanation**
- `total_timesteps` - Total training steps completed
- `mean_reward` - Average reward over last 100 episodes
- `episode_count` - Total episodes completed
- `learning_rate` - Current PPO learning rate
- `hallucination_detection_rate` - % of hallucinations detected
- `average_episode_length` - Average steps per episode

---

### GET /training/metrics/history
Get historical metrics.

**Query Parameters**
- `limit` (integer, optional) - Maximum records to return (default: 100)

**Response (200)**
```json
{
  "metrics": [
    {
      "timestamp": "2026-04-25T12:00:00.000000",
      "total_timesteps": 50000,
      "mean_reward": 52.3,
      "episode_count": 100,
      "hallucination_detection_rate": 0.87
    },
    {
      "timestamp": "2026-04-25T11:50:00.000000",
      "total_timesteps": 45000,
      "mean_reward": 48.1,
      "episode_count": 90,
      "hallucination_detection_rate": 0.82
    }
  ],
  "total_records": 2,
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

---

## Model Endpoints

### GET /model/info
Get model information.

**Response (200)**
```json
{
  "model_name": "ppo_manager",
  "training_timesteps": 50000,
  "hyperparameters": {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2
  },
  "created_at": "2026-04-25T10:00:00.000000"
}
```

---

### POST /model/checkpoint
Save model checkpoint.

**Request Body**
```json
{
  "repo_id": "username/orchestraai-model"
}
```

**Response (200)**
```json
{
  "checkpoint_id": "checkpoint_1234567890",
  "status": "saved",
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

---

## WebSocket Endpoint

### WS /ws/live
Real-time updates via WebSocket.

**Connection**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};

ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'metrics'
}));
```

**Message Types**
- `episode_update` - Episode state changed
- `metrics_update` - Metrics updated
- `error` - Error occurred

**Example Message**
```json
{
  "type": "episode_update",
  "episode_id": "ep_abc123def456",
  "step": 50,
  "reward": 5.2,
  "timestamp": "2026-04-25T12:00:00.000000"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

---

## Data Types

### Observation
```json
{
  "task_embedding": [float, ...],      // 64-dimensional vector
  "worker_states": [[float, ...], ...], // 4x5 matrix
  "subtask_status": [int, ...],         // 4-element binary array
  "budget_remaining": float,            // 0.0 to 1.0
  "steps_remaining": float              // 0.0 to 1.0
}
```

### Episode
```json
{
  "episode_id": "string",
  "status": "active|completed|failed",
  "reward": float,
  "steps": int,
  "created_at": "ISO 8601 datetime",
  "ended_at": "ISO 8601 datetime (optional)"
}
```

### Metrics
```json
{
  "total_timesteps": int,
  "mean_reward": float,
  "episode_count": int,
  "learning_rate": float,
  "hallucination_detection_rate": float,
  "average_episode_length": float,
  "timestamp": "ISO 8601 datetime"
}
```

---

## Rate Limiting

Currently no rate limiting. In production, implement:
- 100 requests per minute per IP
- 1000 requests per hour per API key

---

## Pagination

For endpoints that return lists:
- Default limit: 10
- Maximum limit: 100
- Use `skip` and `limit` query parameters

Example:
```
GET /episodes?skip=10&limit=20
```

---

## Timestamps

All timestamps are in ISO 8601 format (UTC):
```
2026-04-25T12:00:00.000000
```

---

## Example Usage

### Python
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Start episode
response = requests.post(f"{BASE_URL}/episode/start", json={
    "config": {"num_workers": 4, "budget": 1000}
})
episode = response.json()
episode_id = episode["episode_id"]

# Execute action
response = requests.post(f"{BASE_URL}/episode/{episode_id}/step", json={
    "action_id": 0,
    "target_worker_id": 1
})
step_result = response.json()

# Get metrics
response = requests.get(f"{BASE_URL}/training/metrics")
metrics = response.json()

# End episode
response = requests.post(f"{BASE_URL}/episode/{episode_id}/end")
final_result = response.json()
```

### JavaScript
```javascript
const BASE_URL = "http://localhost:8000";

// Start episode
const startResponse = await fetch(`${BASE_URL}/episode/start`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    config: { num_workers: 4, budget: 1000 }
  })
});
const episode = await startResponse.json();
const episodeId = episode.episode_id;

// Execute action
const stepResponse = await fetch(`${BASE_URL}/episode/${episodeId}/step`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    action_id: 0,
    target_worker_id: 1
  })
});
const stepResult = await stepResponse.json();

// Get metrics
const metricsResponse = await fetch(`${BASE_URL}/training/metrics`);
const metrics = await metricsResponse.json();
```

### cURL
```bash
# Start episode
curl -X POST http://localhost:8000/episode/start \
  -H "Content-Type: application/json" \
  -d '{"config": {"num_workers": 4, "budget": 1000}}'

# Execute action
curl -X POST http://localhost:8000/episode/ep_abc123/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 0, "target_worker_id": 1}'

# Get metrics
curl http://localhost:8000/training/metrics

# List episodes
curl http://localhost:8000/episodes
```

---

## Interactive Documentation

Access interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Support

For issues or questions:
1. Check the logs: `tail -f server.log`
2. Test endpoint: `curl http://localhost:8000/health`
3. View API docs: `http://localhost:8000/docs`

---

**API Version**: 1.0.0  
**Last Updated**: April 25, 2026  
**Status**: ✅ Production Ready
