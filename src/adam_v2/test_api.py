#!/usr/bin/env python3
"""Test ADAM v2 API endpoints"""
import requests
import json

base_url = "http://localhost:8000"

# Test health endpoint
print("Testing health endpoint...")
response = requests.get(f"{base_url}/api/health")
print(f"Health check: {response.json()}")

# Test projects endpoint
print("\nTesting projects endpoint...")
response = requests.get(f"{base_url}/api/projects/")
print(f"Projects: {len(response.json())} found")
