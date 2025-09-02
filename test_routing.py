import asyncio
import sys
from pathlib import Path
sys.path.append('/Users/vitoryago/ADAM/src')

from adam_v2.services.fast_routing_service import FastRoutingService

async def test_routing():
    service = FastRoutingService()
    
    test_queries = [
        "Can you develop a python code for poker gaming? Need to be a complex game",
        "Hey ADAM",
        "Explain how neural networks work",
        "I need help with PDT to DBT conversion"
    ]
    
    for query in test_queries:
        result = await service.route_query(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"Model: {result['model']}")
        print(f"Tier: {result['tier']}")

asyncio.run(test_routing())
