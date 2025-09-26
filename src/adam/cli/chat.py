#!/usr/bin/env python3
"""
ADAM Chat CLI - Simple chat interface
"""

import asyncio
from pathlib import Path
from typing import Optional
import os

from adam import ADAMSystem
from adam.config.unified import get_config
from adam.llm.async_client import AsyncLLMClient


def main():
    """Main entry point for adam-chat command"""
    print("ADAM Chat Interface")
    print("=" * 50)

    # Initialize configuration
    config = get_config()

    # Initialize ADAM system
    adam = ADAMSystem(config=config)

    # Initialize LLM client
    llm_client = AsyncLLMClient()

    print("Type 'exit' to quit, 'clear' to clear screen\n")

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            elif not user_input:
                continue

            # Process with ADAM
            print("\nADAM: ", end="", flush=True)

            # Get response from LLM
            try:
                response = asyncio.run(llm_client.complete(user_input))
                if hasattr(response, 'content'):
                    print(response.content)
                else:
                    print(response)
            except Exception as e:
                print(f"Sorry, I couldn't process that request: {e}")
                print("\nTroubleshooting:")
                print("1. Create a .env file in your project root with:")
                print("   XAI_API_KEY=your_xai_api_key_here")
                print("   OPENAI_API_KEY=your_openai_api_key_here")
                print("2. Install missing dependencies: pip install -e . --force-reinstall")
                print("3. Check that your API keys are valid and have credits")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()