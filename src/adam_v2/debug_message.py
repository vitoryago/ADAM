#!/usr/bin/env python3
"""Debug message rendering"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from models import Message

async def check_messages():
    # Create database connection
    engine = create_async_engine("sqlite+aiosqlite:///./data/adam_v2.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Get recent assistant messages
        result = await session.execute(
            select(Message)
            .where(Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(3)
        )
        messages = result.scalars().all()
        
        for i, msg in enumerate(messages):
            print(f"\n{'='*50}")
            print(f"Message {i+1} (ID: {msg.id})")
            print(f"Model: {msg.model}")
            print(f"Cost: ${msg.cost or 0:.4f}")
            print(f"Content preview (first 200 chars):")
            print(msg.content[:200])
            print(f"\nChecking for code blocks:")
            if "```" in msg.content:
                print("✓ Contains code blocks")
            else:
                print("✗ No code blocks found")
    
    await engine.dispose()

asyncio.run(check_messages())