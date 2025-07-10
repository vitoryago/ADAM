#!/usr/bin/env python3
"""Quick ADAM chat - minimal UI, maximum functionality"""
from src.adam.integrated_conversation_system import IntegratedConversationSystem

adam = IntegratedConversationSystem()
session = adam.start_session("Quick Chat")

print("ADAM is ready! (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break
    
    response = adam.process_message(query)
    print(f"\nADAM: {response}\n")
    print(f"[Model: {adam.last_model_used}, Cost: ${adam.last_cost:.4f}]\n")