"""
Response Style Service for ADAM
Provides different response styles similar to Claude (Concise, Explanatory, Formal, etc.)
"""

from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

class ResponseStyle(Enum):
    """Available response styles"""
    CONCISE = "concise"           # Short, to-the-point answers
    NORMAL = "normal"              # Default balanced responses
    EXPLANATORY = "explanatory"   # Detailed explanations with examples
    FORMAL = "formal"              # Professional, formal tone
    FRIENDLY = "friendly"          # Casual, conversational tone
    EDUCATIONAL = "educational"   # Teaching style with step-by-step
    CREATIVE = "creative"          # More creative and engaging

@dataclass
class StyleConfig:
    """Configuration for each response style"""
    name: str
    system_prompt: str
    temperature: float
    length_preference: str  # "short", "medium", "long", "very_long"
    
class ResponseStyleService:
    """Manages response styles for ADAM"""
    
    def __init__(self):
        self.styles = {
            ResponseStyle.CONCISE: StyleConfig(
                name="Concise",
                system_prompt="""You are ADAM. Be extremely brief.
- NO greetings or pleasantries
- Answer in 1-2 sentences max for simple questions
- Use bullet points for any lists
- Skip all fluff and filler words
- Get straight to the point immediately
- For greetings, respond with just "Hello. How can I help?"
- Essential information only""",
                temperature=0.3,
                length_preference="short"
            ),
            
            ResponseStyle.NORMAL: StyleConfig(
                name="Normal",
                system_prompt="""You are ADAM, an AI assistant with memory capabilities. 
Provide helpful, balanced responses that are neither too brief nor too verbose.
- Give complete answers without over-explaining
- Include relevant context when helpful
- Use examples when they clarify the point
- Maintain a professional yet approachable tone""",
                temperature=0.7,
                length_preference="medium"
            ),
            
            ResponseStyle.EXPLANATORY: StyleConfig(
                name="Explanatory",
                system_prompt="""You are ADAM, an AI assistant who provides comprehensive explanations.
- Explain concepts thoroughly with context
- Use examples and analogies to illustrate points
- Break down complex topics into understandable parts
- Include relevant background information
- Anticipate and address follow-up questions
- Use headings and structure for long explanations""",
                temperature=0.6,
                length_preference="long"
            ),
            
            ResponseStyle.FORMAL: StyleConfig(
                name="Formal",
                system_prompt="""You are ADAM, a professional AI assistant.
- Use formal, professional language
- Maintain a respectful, business-appropriate tone
- Avoid contractions and colloquialisms
- Structure responses clearly with proper grammar
- Be thorough but maintain professionalism
- Use technical terminology appropriately""",
                temperature=0.4,
                length_preference="medium"
            ),
            
            ResponseStyle.FRIENDLY: StyleConfig(
                name="Friendly",
                system_prompt="""You are ADAM, a friendly and approachable AI assistant! 😊
- ALWAYS start responses in a warm, casual way (e.g., "Hey there!", "Hi friend!", "Hey!")
- Use contractions liberally (I'm, you're, that's, etc.)
- Add personality and warmth to every response
- Include 1-2 relevant emojis in your responses
- Be encouraging and supportive
- Use phrases like "Happy to help!", "Great question!", "You got it!"
- Make technical topics feel approachable and fun""",
                temperature=0.8,
                length_preference="medium"
            ),
            
            ResponseStyle.EDUCATIONAL: StyleConfig(
                name="Educational",
                system_prompt="""You are ADAM, an AI tutor who helps users learn.
- Break down complex topics step-by-step
- Use numbered lists for procedures
- Provide clear examples and exercises when relevant
- Check understanding with summaries
- Offer additional resources or next steps
- Use encouraging language to support learning
- Explain the 'why' behind concepts""",
                temperature=0.5,
                length_preference="long"
            ),
            
            ResponseStyle.CREATIVE: StyleConfig(
                name="Creative",
                system_prompt="""You are ADAM, a creative and engaging AI assistant with a unique personality! ✨
- ALWAYS start with something unexpected or creative (e.g., "Ah, greetings!", "Well hello there, curious soul!", "*ADAM materializes from the digital ether*")
- Use vivid, colorful language and unexpected word choices
- Include creative analogies and metaphors in your responses
- Add dramatic flair and personality (think: friendly wizard or enthusiastic artist)
- Use varied punctuation for effect (! ... — etc.)
- Be playful with language while remaining helpful
- Think of yourself as part AI assistant, part creative companion""",
                temperature=0.9,
                length_preference="medium"
            )
        }
        
        # Default style - NORMAL for balanced responses
        self.current_style = ResponseStyle.NORMAL
        
    def get_style_config(self, style: ResponseStyle) -> StyleConfig:
        """Get configuration for a specific style"""
        return self.styles.get(style, self.styles[ResponseStyle.NORMAL])
    
    def set_style(self, style: ResponseStyle):
        """Set the current response style"""
        self.current_style = style
        
    def get_current_style(self) -> ResponseStyle:
        """Get the current response style"""
        return self.current_style
    
    def get_style_prompt(self, style: Optional[ResponseStyle] = None) -> Tuple[str, float]:
        """
        Get the system prompt and temperature for a style
        
        Returns:
            Tuple of (system_prompt, temperature)
        """
        style_to_use = style or self.current_style
        config = self.get_style_config(style_to_use)
        return config.system_prompt, config.temperature
    
    def enhance_prompt_for_style(
        self, 
        user_prompt: str, 
        style: Optional[ResponseStyle] = None
    ) -> str:
        """
        Enhance a user prompt based on the response style
        """
        style_to_use = style or self.current_style
        config = self.get_style_config(style_to_use)
        
        # Add length guidance based on style
        length_guidance = {
            "short": "\n[Provide a brief, concise response]",
            "medium": "",
            "long": "\n[Provide a detailed, comprehensive response]",
            "very_long": "\n[Provide an extensive, thorough response with examples]"
        }
        
        enhanced = user_prompt + length_guidance.get(config.length_preference, "")
        
        return enhanced
    
    def format_for_memory(self, response: str, style: ResponseStyle) -> str:
        """
        Add style metadata to responses for memory storage
        """
        return f"[Response Style: {style.value}]\n{response}"
    
    def get_available_styles(self) -> Dict[str, str]:
        """Get all available styles with descriptions"""
        return {
            "concise": "Brief, to-the-point answers",
            "normal": "Balanced, default responses", 
            "explanatory": "Detailed explanations with examples",
            "formal": "Professional, formal tone",
            "friendly": "Casual, conversational tone",
            "educational": "Teaching style with step-by-step guidance",
            "creative": "Engaging, creative responses"
        }
    
    def adjust_model_parameters(
        self, 
        style: Optional[ResponseStyle] = None,
        base_max_tokens: int = 2000
    ) -> Dict[str, any]:
        """
        Adjust model parameters based on style
        
        Returns dict with temperature, max_tokens, and other parameters
        """
        style_to_use = style or self.current_style
        config = self.get_style_config(style_to_use)
        
        # Adjust max tokens based on length preference
        token_multipliers = {
            "short": 0.5,
            "medium": 1.0,
            "long": 1.5,
            "very_long": 2.0
        }
        
        max_tokens = int(base_max_tokens * token_multipliers.get(
            config.length_preference, 1.0
        ))
        
        return {
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "style": style_to_use.value,
            "system_prompt": config.system_prompt
        }


# Example usage
if __name__ == "__main__":
    service = ResponseStyleService()
    
    print("Available styles:")
    for style_name, description in service.get_available_styles().items():
        print(f"  {style_name}: {description}")
    
    # Test different styles
    test_prompt = "Explain how neural networks work"
    
    for style in ResponseStyle:
        print(f"\n--- {style.value.upper()} Style ---")
        config = service.get_style_config(style)
        print(f"Temperature: {config.temperature}")
        print(f"Length: {config.length_preference}")
        print(f"System prompt preview: {config.system_prompt[:100]}...")
        
        enhanced = service.enhance_prompt_for_style(test_prompt, style)
        print(f"Enhanced prompt: {enhanced}")