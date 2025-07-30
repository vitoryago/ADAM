# ADAM Frontend Integration Guide

## Overview

This frontend is now configured to work with the ADAM v2 backend API running on port 8000.

## Architecture

```
Frontend (React + Vite) --> ADAM Backend API
   Port 5173                    Port 8000
```

## Quick Start

### 1. Start ADAM Backend (if not running)
```bash
cd /Users/vitoryago/ADAM/src/adam_v2
/Users/vitoryago/ADAM/venv/bin/python main.py
```

### 2. Start Frontend
```bash
cd /Users/vitoryago/ADAM/frontend/AdamChat
./start-frontend.sh
```

### 3. Access the Application
Open http://localhost:5173 in your browser

## API Endpoints

The frontend will communicate with these ADAM backend endpoints:

- `GET /api/health` - Health check
- `GET /api/projects/` - List projects
- `POST /api/projects/` - Create project
- `GET /api/projects/{id}/conversations` - List conversations
- `POST /api/conversations/{id}/messages` - Send message
- `POST /api/conversations/{id}/messages/stream` - Stream response

## Configuration

- Frontend proxy is configured in `vite.config.ts`
- All `/api/*` requests are forwarded to `http://localhost:8000`
- WebSocket connections `/ws/*` are forwarded to `ws://localhost:8000`

## Troubleshooting

1. **CORS Issues**: The backend is already configured to accept requests from localhost:5173
2. **API Connection Failed**: Ensure the backend is running on port 8000
3. **Missing Dependencies**: Run `npm install` in the frontend directory

## Development Notes

- The frontend uses React Query for API state management
- API requests are made through the `apiRequest` utility in `client/src/lib/queryClient.ts`
- The Node.js server in `server/` is optional - you can use Vite dev server directly