#!/usr/bin/env python3
"""
Test script for lineage and onboarding features
"""

import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8000/api"

def test_lineage():
    """Test lineage analysis"""
    print("Testing Lineage Analysis...")
    
    # Analyze the ADAM project
    response = requests.post(
        f"{API_BASE}/lineage/analyze",
        params={
            "path": "/Users/vitoryago/ADAM/src/adam",
            "pattern": "*.py"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Lineage analysis successful!")
        print(f"  - Files analyzed: {result['result']['files_analyzed']}")
        print(f"  - Nodes found: {result['result']['nodes']}")
        print(f"  - Edges found: {result['result']['edges']}")
        return True
    else:
        print(f"❌ Lineage analysis failed: {response.text}")
        return False

def test_onboarding():
    """Test onboarding path creation"""
    print("\nTesting Onboarding Path Creation...")
    
    # Create an onboarding path
    response = requests.post(
        f"{API_BASE}/onboarding/create-path",
        json={
            "project_path": "/Users/vitoryago/ADAM",
            "user_level": "intermediate",
            "focus_area": "src/adam",
            "custom_requirements": "Focus on understanding the memory system and LLM integration"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Onboarding path created!")
        print(f"  - Path ID: {result['path_id']}")
        print(f"  - Project: {result['project_name']}")
        print(f"  - Milestones: {result['milestones']}")
        print(f"  - Estimated time: {result['estimated_time']} minutes")
        
        # Display first milestone
        if result['data']['milestones']:
            first_milestone = result['data']['milestones'][0]
            print(f"\n  First Milestone: {first_milestone['title']}")
            print(f"  - {first_milestone['description']}")
            print(f"  - Phase: {first_milestone['phase']}")
            print(f"  - Tasks: {len(first_milestone['tasks'])}")
        
        return result['path_id']
    else:
        print(f"❌ Onboarding creation failed: {response.text}")
        return None

def test_recommendation(path_id):
    """Test AI recommendation"""
    print(f"\nTesting AI Recommendation for path {path_id}...")
    
    response = requests.get(
        f"{API_BASE}/onboarding/path/{path_id}/recommendation"
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Got recommendation!")
        message = str(result.get('message', 'No message'))
        print(f"\n  ADAM says: {message[:200]}...")
        return True
    else:
        print(f"❌ Recommendation failed: {response.text}")
        return False

def main():
    print("=" * 60)
    print("ADAM Lineage and Onboarding Test Suite")
    print("=" * 60)
    
    # Test lineage
    lineage_ok = test_lineage()
    
    # Test onboarding
    path_id = test_onboarding()
    
    # Test recommendation if path was created
    if path_id:
        rec_ok = test_recommendation(path_id)
    else:
        rec_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  - Lineage Analysis: {'✅ PASSED' if lineage_ok else '❌ FAILED'}")
    print(f"  - Onboarding Creation: {'✅ PASSED' if path_id else '❌ FAILED'}")
    print(f"  - AI Recommendation: {'✅ PASSED' if rec_ok else '❌ FAILED'}")
    print("=" * 60)
    
    if lineage_ok and path_id and rec_ok:
        print("\n🎉 All tests passed! The lineage and onboarding features are working.")
        print(f"\nYou can now:")
        print(f"1. Open http://localhost:8501 to use the Onboarding UI")
        print(f"2. View API docs at http://localhost:8000/api/docs")
        print(f"3. Start onboarding with path ID: {path_id}")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()