#!/usr/bin/env python3
"""Check latest message content"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from models import Message

async def check_latest():
    engine = create_async_engine("sqlite+aiosqlite:///./data/adam_v2.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Get the latest assistant message
        result = await session.execute(
            select(Message)
            .where(Message.role == "assistant")
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        msg = result.scalar_one_or_none()
        
        if msg:
            print(f"Message ID: {msg.id}")
            print(f"Model: {msg.model}")
            print(f"Created: {msg.created_at}")
            print(f"\nContent (first 500 chars):")
            print(msg.content[:500])
            print(f"\n\nChecking for issues:")
            if "[object Object]" in msg.content:
                print("⚠️  Found [object Object] in content!")
            if "```" in msg.content:
                print("✓ Contains code blocks")
                # Find first code block
                start = msg.content.find("```")
                end = msg.content.find("```", start + 3)
                if start != -1 and end != -1:
                    print(f"\nFirst code block:")
                    print(msg.content[start:end+3])
    
    await engine.dispose()

asyncio.run(check_latest())