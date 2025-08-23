"""
Feedback Analysis API endpoint for intelligent user feedback understanding
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import json
import logging
import asyncio

from adam.llm.client import UnifiedLLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["feedback"])

class FeedbackAnalysisRequest(BaseModel):
    prompt: str
    
class FeedbackAnalysisResponse(BaseModel):
    analysis: str

class FeedbackAnalyzer:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
    
    async def analyze_feedback(self, prompt: str) -> str:
        """
        Use LLM to analyze user feedback and understand intent
        """
        try:
            # Add clear instruction for JSON output
            enhanced_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON, no additional text."
            
            # Use the LLM to analyze feedback
            # Using gpt-5-mini for fast, cheap feedback analysis
            response = await self.llm_client.complete(
                prompt=enhanced_prompt,
                model="gpt-5-mini",  # GPT-5 Mini - cheapest and fastest for simple analysis
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000  # Increased for complete JSON responses
            )
            
            # Extract the content from LLMResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Ensure response is valid JSON
            try:
                # Try to parse as JSON to validate
                json.loads(response_text)
                return response_text
            except json.JSONDecodeError as e:
                # Log the actual response for debugging
                logger.warning(f"LLM response was not valid JSON: {response_text[:500]}")
                # If not valid JSON, create a structured response
                return json.dumps({
                    "type": "unclear",
                    "confidence": 0.5,
                    "isAboutLastError": False,
                    "providedSolution": None,
                    "errorDescription": None,
                    "reasoning": f"Could not parse LLM response: {str(e)}"
                })
                
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")
            # Return a default response on error
            return json.dumps({
                "type": "unclear",
                "confidence": 0.3,
                "isAboutLastError": False,
                "providedSolution": None,
                "errorDescription": None,
                "reasoning": f"Analysis error: {str(e)}"
            })

# Create global analyzer instance
feedback_analyzer = FeedbackAnalyzer()

@router.post("/analyze-feedback", response_model=FeedbackAnalysisResponse)
async def analyze_feedback(request: FeedbackAnalysisRequest) -> FeedbackAnalysisResponse:
    """
    Analyze user feedback using LLM to understand intent
    """
    try:
        analysis = await feedback_analyzer.analyze_feedback(request.prompt)
        return FeedbackAnalysisResponse(analysis=analysis)
    except Exception as e:
        logger.error(f"Feedback analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}