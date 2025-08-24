#!/usr/bin/env python3
"""
Create the default ADAM project if it doesn't exist
"""
import requests
import json

# The project ID used by the VSCode extension
PROJECT_ID = "3a859e97-16fd-46c6-b018-1ede9fade704"

# Check if project exists
response = requests.get(f"http://localhost:8000/api/projects/{PROJECT_ID}")

if response.status_code == 404:
    print("Project doesn't exist, creating...")
    
    # Create the project
    create_response = requests.post(
        "http://localhost:8000/api/projects",
        json={
            "name": "VSCode Workspace",
            "description": "Default project for VSCode extension",
            "settings": {
                "use_memory": True,
                "model_preference": "automatic"
            }
        }
    )
    
    if create_response.status_code == 201:
        project = create_response.json()
        print(f"Created project: {project['id']}")
        
        # Now update it to have the specific ID we need
        # (This might not work if the backend doesn't allow ID updates)
        print(f"Note: Project created with ID {project['id']}")
        print(f"You may need to update the extension to use this ID instead of {PROJECT_ID}")
    else:
        print(f"Failed to create project: {create_response.text}")
else:
    print(f"Project {PROJECT_ID} already exists!")
    print(response.json())