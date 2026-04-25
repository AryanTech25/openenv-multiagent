# Complete OrchestraAI API Guide

## 📚 Documentation Files

1. **ORCHESTRAAI_API_DOCS.md** - Full API documentation with all endpoints
2. **API_QUICK_REFERENCE.md** - Quick reference for common tasks
3. **COMPLETE_API_GUIDE.md** - This file

---

## 🚀 Getting Started

### 1. Start the Server
```bash
cd manager-worker-env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 2. Verify Server is Running
```bash
curl http://localhost:8000/health
```

### 3. Access API Documentation
- **Interactive**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📋 API Overview

### Base URL
```
http://localhost:8000
```

### Total Endpoints: 14

**Categories:**
- Root & Health: 3 endpoints
- Episodes: 6 endpoints
- Metrics: 2 endpoints
- Model: 2 endpoints
- WebSocket: 1 endpoint

---

## 🔄 Typical Workflow

### 1. Check Server Health
```bash
curl http://localhost:8000/health
```

### 2. Start an Episode
```bash
curl -X POST http://localhost:8000/episode/start \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "num_workers": 4,
      "budget": 1000,
      "max_steps": 500
    }
  }'
```

Response:
```json
{
  "episode_id": "ep_abc123def456",
  "observation": {...},
  "created_at": "2026-04-25T12:00:00.000000"
}
```

### 3. Execute Actions
```bash
curl -X POST http://localhost:8000/episode/ep_abc123def456/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_id": 0,
    "target_worker_id": 1
  }'
```

Response:
```json
{
  "observation": {...},
  "reward": 5.2,
  "done": false,
  "info": {...},
  "step_number": 1
}
```

### 4. Monitor Metrics
```bash
curl http://localhost:8000/training/metrics
```

Response:
```json
{
  "total_timesteps": 50000,
  "mean_reward": 52.3,
  "episode_count": 100,
  "hallucination_detection_rate": 0.87,
  ...
}
```

### 5. End Episode
```bash
curl -X POST http://localhost:8000/episode/ep_abc123def456/end
```

Response:
```json
{
  "episode_id": "ep_abc123def456",
  "final_reward": 125.5,
  "final_quality": 0.92,
  "total_steps": 150,
  ...
}
```

---

## 🎯 Common Tasks

### Get Episode Status
```bash
curl http://localhost:8000/episode/ep_abc123def456
```

### List All Episodes
```bash
curl http://localhost:8000/episodes
```

### Get Episode History
```bash
curl http://localhost:8000/episode/ep_abc123def456/history
```

### Get Metrics History
```bash
curl http://localhost:8000/training/metrics/history?limit=50
```

### Get Model Info
```bash
curl http://localhost:8000/model/info
```

### Save Model Checkpoint
```bash
curl -X POST http://localhost:8000/model/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "username/orchestraai-model"}'
```

---

## 💻 Code Examples

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
print(f"Started episode: {episode_id}")

# Execute 10 actions
for i in range(10):
    response = requests.post(f"{BASE_URL}/episode/{episode_id}/step", json={
        "action_id": i % 7,
        "target_worker_id": i % 4
    })
    result = response.json()
    print(f"Step {i+1}: Reward={result['reward']:.2f}, Done={result['done']}")

# Get metrics
response = requests.get(f"{BASE_URL}/training/metrics")
metrics = response.json()
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Hallucination Detection: {metrics['hallucination_detection_rate']:.2%}")

# End episode
response = requests.post(f"{BASE_URL}/episode/{episode_id}/end")
final = response.json()
print(f"Final Reward: {final['final_reward']:.2f}")
print(f"Final Quality: {final['final_quality']:.2f}")
```

### JavaScript
```javascript
const BASE_URL = "http://localhost:8000";

async function runEpisode() {
  // Start episode
  const startRes = await fetch(`${BASE_URL}/episode/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config: { num_workers: 4, budget: 1000 }
    })
  });
  const episode = await startRes.json();
  const episodeId = episode.episode_id;
  console.log(`Started episode: ${episodeId}`);

  // Execute 10 actions
  for (let i = 0; i < 10; i++) {
    const stepRes = await fetch(`${BASE_URL}/episode/${episodeId}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action_id: i % 7,
        target_worker_id: i % 4
      })
    });
    const result = await stepRes.json();
    console.log(`Step ${i+1}: Reward=${result.reward.toFixed(2)}, Done=${result.done}`);
  }

  // Get metrics
  const metricsRes = await fetch(`${BASE_URL}/training/metrics`);
  const metrics = await metricsRes.json();
  console.log(`Mean Reward: ${metrics.mean_reward.toFixed(2)}`);
  console.log(`Hallucination Detection: ${(metrics.hallucination_detection_rate * 100).toFixed(1)}%`);

  // End episode
  const endRes = await fetch(`${BASE_URL}/episode/${episodeId}/end`, {
    method: "POST"
  });
  const final = await endRes.json();
  console.log(`Final Reward: ${final.final_reward.toFixed(2)}`);
  console.log(`Final Quality: ${final.final_quality.toFixed(2)}`);
}

runEpisode();
```

### cURL Script
```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

# Start episode
echo "Starting episode..."
RESPONSE=$(curl -s -X POST $BASE_URL/episode/start \
  -H "Content-Type: application/json" \
  -d '{"config": {"num_workers": 4, "budget": 1000}}')
EPISODE_ID=$(echo $RESPONSE | jq -r '.episode_id')
echo "Episode ID: $EPISODE_ID"

# Execute 10 actions
for i in {1..10}; do
  ACTION_ID=$((($i - 1) % 7))
  WORKER_ID=$((($i - 1) % 4))
  
  RESPONSE=$(curl -s -X POST $BASE_URL/episode/$EPISODE_ID/step \
    -H "Content-Type: application/json" \
    -d "{\"action_id\": $ACTION_ID, \"target_worker_id\": $WORKER_ID}")
  
  REWARD=$(echo $RESPONSE | jq '.reward')
  DONE=$(echo $RESPONSE | jq '.done')
  echo "Step $i: Reward=$REWARD, Done=$DONE"
done

# Get metrics
echo "Getting metrics..."
curl -s $BASE_URL/training/metrics | jq '.'

# End episode
echo "Ending episode..."
curl -s -X POST $BASE_URL/episode/$EPISODE_ID/end | jq '.'
```

---

## 🔌 WebSocket Usage

### JavaScript
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onopen = () => {
  console.log('Connected to WebSocket');
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'metrics'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from WebSocket');
};
```

### Python
```python
import asyncio
import websockets
import json

async def listen_updates():
    uri = "ws://localhost:8000/ws/live"
    async with websockets.connect(uri) as websocket:
        # Subscribe to metrics
        await websocket.send(json.dumps({
            "type": "subscribe",
            "channel": "metrics"
        }))
        
        # Listen for updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Update: {data}")

asyncio.run(listen_updates())
```

---

## 📊 Data Structures

### Episode Configuration
```json
{
  "num_workers": 4,
  "budget": 1000,
  "max_steps": 500,
  "difficulty": 3
}
```

### Observation
```json
{
  "task_embedding": [-0.5, 0.3, ...],
  "worker_states": [[0.8, 0.2, ...], ...],
  "subtask_status": [0, 0, 1, 0],
  "budget_remaining": 0.85,
  "steps_remaining": 0.9
}
```

### Action
```json
{
  "action_id": 0,
  "target_worker_id": 1
}
```

### Step Result
```json
{
  "observation": {...},
  "reward": 5.2,
  "done": false,
  "info": {
    "action_cost": 10,
    "worker_quality": 0.85,
    "hallucination_detected": false
  },
  "step_number": 1
}
```

---

## ⚠️ Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "detail": "Invalid action_id. Must be 0-6"
}
```

**404 Not Found**
```json
{
  "detail": "Episode not found"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Internal server error"
}
```

### Error Handling in Python
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/episode/invalid_id/step",
        json={"action_id": 0}
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Details: {e.response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

---

## 🔐 Security Considerations

### Current State
- No authentication required
- No rate limiting
- No CORS restrictions

### Production Recommendations
1. Add API key authentication
2. Implement rate limiting (100 req/min)
3. Enable HTTPS
4. Add request validation
5. Implement logging and monitoring
6. Add request signing

---

## 📈 Performance Tips

1. **Batch Operations**: Group multiple actions in a single request
2. **Pagination**: Use `skip` and `limit` for large result sets
3. **Caching**: Cache metrics that don't change frequently
4. **Connection Pooling**: Reuse HTTP connections
5. **WebSocket**: Use WebSocket for real-time updates instead of polling

---

## 🐛 Debugging

### Check Server Status
```bash
curl http://localhost:8000/health
```

### View API Documentation
```
http://localhost:8000/docs
```

### Check Logs
```bash
# If running in terminal, logs appear there
# If running in background, check log file
tail -f server.log
```

### Test Endpoint
```bash
curl -v http://localhost:8000/health
```

---

## 📞 Support

### Resources
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- GitHub: `https://github.com/yourusername/orchestraai`
- Issues: `https://github.com/yourusername/orchestraai/issues`

### Troubleshooting
1. Verify server is running: `curl http://localhost:8000/health`
2. Check MongoDB connection: See `MONGODB_FASTAPI_COMPLETE.md`
3. Review logs for errors
4. Test with simple endpoint first

---

## 📝 Summary

**OrchestraAI API** provides a complete interface for:
- ✅ Managing episodes
- ✅ Executing actions
- ✅ Tracking metrics
- ✅ Monitoring models
- ✅ Real-time updates

**Status**: 🟢 Production Ready

---

**Version**: 1.0.0  
**Last Updated**: April 25, 2026  
**Maintained By**: OrchestraAI Team
