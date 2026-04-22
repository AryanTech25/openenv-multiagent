# Backend API - Manager-Worker RL Environment

FastAPI backend server for the Manager-Worker RL Environment with MongoDB integration.

## Features

- ✅ RESTful API for episode management
- ✅ WebSocket for real-time updates
- ✅ Training metrics tracking
- ✅ MongoDB integration
- ✅ CORS support
- ✅ Comprehensive error handling
- ✅ Async/await for scalability

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn motor pydantic-settings python-dotenv
```

### 2. Configure MongoDB

Create a `.env` file in the BE folder:

```bash
cp .env.example .env
```

Then edit `.env` and add your MongoDB connection string:

```
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

### 3. Run the Server

```bash
cd manager-worker-env/BE
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health & Status

- `GET /health` - Health check
- `GET /status` - Server status
- `GET /` - Root endpoint

### Episode Management

- `POST /episode/start` - Start new episode
- `POST /episode/{episode_id}/step` - Execute action
- `GET /episode/{episode_id}/state` - Get episode state
- `GET /episode/{episode_id}/history` - Get episode history
- `POST /episode/{episode_id}/reset` - Reset episode
- `POST /episode/{episode_id}/end` - End episode
- `GET /episode/list` - List active episodes

### Training Metrics

- `GET /training/metrics` - Current metrics
- `GET /training/metrics/history` - Historical metrics
- `GET /training/model/info` - Model information
- `GET /training/model/checkpoint` - Checkpoint info
- `POST /training/model/save` - Save model to Hub

### WebSocket

- `WS /ws/live` - Real-time updates

## Usage Examples

### Start Episode

```bash
curl -X POST http://localhost:8000/episode/start \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{
  "episode_id": "ep_abc123",
  "observation": {...},
  "created_at": "2026-04-22T10:00:00"
}
```

### Execute Action

```bash
curl -X POST http://localhost:8000/episode/ep_abc123/step \
  -H "Content-Type: application/json" \
  -d '{"action_id": 0}'
```

### Get Metrics

```bash
curl http://localhost:8000/training/metrics
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live');

ws.onopen = () => {
  // Subscribe to episode updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    episode_id: 'ep_abc123'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Update:', message);
};
```

## Project Structure

```
BE/
├── main.py                    (FastAPI app)
├── config.py                  (Configuration)
├── database.py                (MongoDB connection)
├── models.py                  (Pydantic models)
├── episode_manager.py         (Episode lifecycle)
├── websocket_manager.py       (WebSocket management)
├── metrics_tracker.py         (Metrics tracking)
├── errors.py                  (Error handling)
├── __init__.py
├── .env.example               (Environment template)
└── README.md                  (This file)
```

## Configuration

Edit `config.py` to customize:

- CORS origins
- MongoDB settings
- Max episodes
- WebSocket settings
- Logging level

## Error Handling

All errors return standardized JSON responses:

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable message",
  "details": "Additional context",
  "timestamp": "2026-04-22T10:00:00"
}
```

## Database

MongoDB collections:

- `episodes` - Episode data
- `metrics` - Training metrics
- `training_sessions` - Training session info

## Performance

- Async/await for non-blocking I/O
- Connection pooling
- Message broadcasting
- Efficient indexing

## Next Steps

1. Provide MongoDB connection string
2. Run the server
3. Test endpoints with curl or Postman
4. Connect React frontend via WebSocket

## Support

For issues or questions, check the main project README.
