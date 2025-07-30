# ADAM Frontend

This directory contains the frontend application for ADAM.

## Integration with Backend

The frontend connects to the ADAM backend API running at:
- Development: http://localhost:8000/api
- API Documentation: http://localhost:8000/api/docs

## Setup Instructions

1. Place your frontend files in this directory
2. Update the API endpoint configuration to point to the backend
3. Install dependencies (if React/Vue/etc.)
4. Run the frontend development server

## CORS Configuration

The backend is already configured to accept requests from:
- http://localhost:3000 (React default)
- http://localhost:5173 (Vite default)

If your frontend runs on a different port, update the CORS settings in `src/adam_v2/main.py`.