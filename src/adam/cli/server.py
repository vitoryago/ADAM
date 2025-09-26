#!/usr/bin/env python3
"""
ADAM Server CLI - FastAPI server wrapper
"""

import sys
import uvicorn
from pathlib import Path

from adam.config.unified import get_config


def main():
    """Main entry point for adam-server command"""
    print("🚀 Starting ADAM FastAPI Server...")
    print("=" * 50)

    # Get configuration
    config = get_config()

    print(f"Host: {config.web.api_host}")
    print(f"Port: {config.web.api_port}")
    print(f"Debug: {config.web.api_debug}")
    print(f"Reload: {config.web.api_reload}")
    print("")

    # Check if adam_v2.main exists and import it
    try:
        # Add the src directory to path temporarily
        src_path = Path(__file__).parent.parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Try to import and run the existing server
        from adam_v2.main import app

        print("✅ Found ADAM v2 server, starting...")

        uvicorn.run(
            "adam_v2.main:app",
            host=config.web.api_host,
            port=config.web.api_port,
            reload=config.web.api_reload,
            log_level="info" if config.core['debug'] else "warning"
        )

    except ImportError as e:
        print(f"⚠️  ADAM v2 server not available: {e}")
        print("Starting basic FastAPI server instead...")

        # Create a basic FastAPI app
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(
            title="ADAM API",
            description="Analytics Data Assistant with Memory",
            version="2.0.0"
        )

        @app.get("/")
        async def root():
            return JSONResponse({
                "message": "ADAM API Server",
                "version": "2.0.0",
                "status": "running"
            })

        @app.get("/health")
        async def health_check():
            return JSONResponse({"status": "healthy"})

        uvicorn.run(
            app,
            host=config.web.api_host,
            port=config.web.api_port,
            log_level="info"
        )


if __name__ == "__main__":
    main()