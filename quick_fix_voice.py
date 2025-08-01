#!/usr/bin/env python3
"""Quick fix for voice service - add debug logging"""

import os
import sys

# Add debug logging to voice service
voice_service_path = "/Users/vitoryago/ADAM/src/adam_v2/services/voice_service.py"

# Read the file
with open(voice_service_path, 'r') as f:
    content = f.read()

# Add import for dotenv if not present
if "from dotenv import load_dotenv" not in content:
    imports = """import os
import asyncio
import logging
from typing import Optional, Union, AsyncGenerator, List, Dict
from dataclasses import dataclass
import base64
import httpx
from enum import Enum
import io
import tempfile
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('.env.local')  # Override with local settings

logger = logging.getLogger(__name__)"""

    content = content.replace("""import os
import asyncio
import logging
from typing import Optional, Union, AsyncGenerator, List, Dict
from dataclasses import dataclass
import base64
import httpx
from enum import Enum
import io
import tempfile
import subprocess

logger = logging.getLogger(__name__)""", imports)

# Save the updated file
with open(voice_service_path, 'w') as f:
    f.write(content)

print("Fixed voice service to load environment variables")