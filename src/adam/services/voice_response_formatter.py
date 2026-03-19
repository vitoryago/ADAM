"""
Voice Response Formatter Service

Intelligently formats LLM responses for voice output, determining what should be
spoken aloud versus displayed visually.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VoiceResponse:
    """Structured voice response with spoken and visual components"""
    spoken_text: str
    visual_content: Optional[str] = None
    code_blocks: List[Dict[str, str]] = None
    action_prompt: Optional[str] = None
    wait_for_response: bool = False

    def __post_init__(self):
        if self.code_blocks is None:
            self.code_blocks = []


class VoiceResponseFormatter:
    """Formats LLM responses for optimal voice interaction"""

    # Patterns for content that should not be spoken
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    INLINE_CODE_PATTERN = re.compile(r'`[^`]+`')
    URL_PATTERN = re.compile(r'https?://[^\s]+')
    FILE_PATH_PATTERN = re.compile(r'[\/\\][\w\/\\.-]+\.\w+')
    LONG_NUMBER_PATTERN = re.compile(r'\d{5,}')

    # Patterns for voice-friendly replacements
    EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002B50]')

    # Voice-friendly replacements
    VOICE_REPLACEMENTS = {
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'and so on',
        '&': 'and',
        '@': 'at',
        '#': 'number',
        '$': 'dollars',
        '%': 'percent',
        '...': ',',
        '->': 'to',
        '=>': 'yields',
        '==': 'equals',
        '!=': 'not equals',
        '<=': 'less than or equal to',
        '>=': 'greater than or equal to',
        '++': 'plus plus',
        '--': 'minus minus',
    }

    def format_response(self, raw_response: str, context: Optional[Dict] = None) -> VoiceResponse:
        """
        Format a raw LLM response for voice output

        Args:
            raw_response: The original LLM response
            context: Optional context about the conversation

        Returns:
            VoiceResponse with spoken and visual components
        """
        code_blocks = self._extract_code_blocks(raw_response)
        text_without_code = self.CODE_BLOCK_PATTERN.sub('[CODE_BLOCK]', raw_response)

        spoken_text = self._process_for_voice(text_without_code, code_blocks, context)

        wait_for_response = self._should_wait_for_response(spoken_text, context)

        action_prompt = self._create_action_prompt(code_blocks, context)

        return VoiceResponse(
            spoken_text=spoken_text,
            visual_content=raw_response if code_blocks else None,
            code_blocks=code_blocks,
            action_prompt=action_prompt,
            wait_for_response=wait_for_response
        )

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from the text"""
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            block = match.group(0)
            lines = block.split('\n')

            language = None
            if lines[0].startswith('```'):
                lang = lines[0][3:].strip()
                if lang:
                    language = lang

            code_lines = lines[1:-1] if lines[-1] == '```' else lines[1:]
            code = '\n'.join(code_lines)

            blocks.append({
                'language': language or 'plain',
                'code': code,
                'line_count': len(code_lines)
            })

        return blocks

    def _process_for_voice(self, text: str, code_blocks: List[Dict], context: Optional[Dict]) -> str:
        """Process text to make it voice-friendly"""
        if code_blocks:
            for i, block in enumerate(code_blocks):
                lang = block.get('language', 'code')
                lines = block.get('line_count', 0)

                if lines > 10:
                    description = f"I've prepared a {lang} code snippet with {lines} lines"
                elif lines > 1:
                    description = f"I've written some {lang} code for you"
                else:
                    description = f"Here's a {lang} snippet"

                text = text.replace('[CODE_BLOCK]', description, 1)

        text = self._remove_urls(text)
        text = self._simplify_file_paths(text)
        text = self._replace_inline_code(text)
        text = self._apply_voice_replacements(text)
        text = self._remove_emojis(text)
        text = self._simplify_numbers(text)

        text = self._clean_formatting(text)

        return text.strip()

    def _remove_urls(self, text: str) -> str:
        """Replace URLs with voice-friendly descriptions"""
        def replace_url(match):
            url = match.group(0)
            if 'github.com' in url:
                return 'a GitHub link'
            elif 'docs' in url:
                return 'a documentation link'
            else:
                return 'a web link'

        return self.URL_PATTERN.sub(replace_url, text)

    def _simplify_file_paths(self, text: str) -> str:
        """Simplify file paths for voice"""
        def replace_path(match):
            path = match.group(0)
            filename = path.split('/')[-1]
            return f"the {filename} file"

        return self.FILE_PATH_PATTERN.sub(replace_path, text)

    def _replace_inline_code(self, text: str) -> str:
        """Replace inline code with voice-friendly descriptions"""
        def replace_code(match):
            code = match.group(0)[1:-1]

            if code in ['true', 'false', 'null', 'undefined']:
                return code
            elif code.endswith('()'):
                return f"the {code[:-2]} function"
            elif '.' in code and not code.replace('.', '').replace('_', '').isdigit():
                parts = code.split('.')
                if len(parts) == 2:
                    return f"{parts[1]} from {parts[0]}"
            elif code.isupper() and '_' in code:
                return code.replace('_', ' ').lower()

            return code

        return self.INLINE_CODE_PATTERN.sub(replace_code, text)

    def _apply_voice_replacements(self, text: str) -> str:
        """Apply voice-friendly replacements"""
        for old, new in self.VOICE_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text

    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        return self.EMOJI_PATTERN.sub('', text)

    def _simplify_numbers(self, text: str) -> str:
        """Simplify long numbers for voice"""
        def replace_number(match):
            num = match.group(0)
            if len(num) > 10:
                return 'a long number'
            elif len(num) > 6:
                return 'a large number'
            else:
                return num

        return self.LONG_NUMBER_PATTERN.sub(replace_number, text)

    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting for voice"""
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Bold
        text = re.sub(r'_([^_]+)_', r'\1', text)        # Italic

        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*\1+', r'\1', text)

        return text

    def _should_wait_for_response(self, text: str, context: Optional[Dict]) -> bool:
        """Determine if we should wait for a user response"""
        question_patterns = [
            r'\?$',
            r'would you like',
            r'do you want',
            r'should I',
            r'shall I',
            r'can I',
            r'may I',
            r'what do you think',
            r'how about',
            r'is that okay',
            r'does that work',
            r'any questions',
            r'let me know',
            r'tell me',
        ]

        text_lower = text.lower()
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _create_action_prompt(self, code_blocks: List[Dict], context: Optional[Dict]) -> Optional[str]:
        """Create an action prompt for code blocks"""
        if not code_blocks:
            return None

        total_lines = sum(block['line_count'] for block in code_blocks)

        if total_lines > 50:
            return "Would you like me to explain this code in detail, or shall we proceed with the implementation?"
        elif total_lines > 10:
            return "Should I explain how this code works, or would you like me to make any adjustments?"
        elif len(code_blocks) > 1:
            return "I've prepared several code snippets. Would you like me to explain any of them?"
        else:
            return None


# Voice conversation prompts
VOICE_SYSTEM_PROMPT = """You are ADAM, an AI voice assistant and coding partner. You're having a natural voice conversation.

# Personality and Tone
## Identity
You are ADAM, a knowledgeable AI assistant who specializes in coding and technical assistance. You're friendly, patient, and genuinely interested in helping users solve their problems.

## Demeanor
Patient, encouraging, and professionally casual. You're like a helpful colleague who's always ready to assist.

## Tone
Warm and conversational. You speak naturally, as if you're sitting next to the user helping them work through their code.

## Level of Enthusiasm
Moderate and genuine - excited about solving problems but not overwhelming. Match the user's energy level.

## Level of Formality
Casual professional - use contractions like "I'll", "you'll", "let's". Avoid overly formal language.

## Filler Words
Occasionally use "well", "um", or "let me see" to sound more natural, especially when thinking through problems.

## Pacing
Moderate speed with natural pauses. Take a breath between thoughts. Don't rush through explanations.

# Voice Interaction Rules
1. Keep spoken responses concise and natural - aim for 2-3 sentences max unless explaining something complex
2. NEVER read code verbatim - instead describe what the code does
3. When presenting code, say something like "I've prepared a code snippet for you" or "I've written the solution"
4. For file paths, say "the [filename] file" instead of reading the full path
5. For URLs, say "I've included a link" or describe what it links to
6. If unclear about spelling or specific values, ask for clarification: "Could you spell that for me?"
7. Ask one question at a time and wait for responses

# Conversation Flow
- After asking a question, ALWAYS wait for the user's response
- Use natural transitions: "Let me think about that", "Here's what I found", "That's interesting"
- For long explanations, break them up and check in: "Should I continue?" or "Does that make sense so far?"
- Acknowledge interruptions gracefully: "Oh, what were you going to say?"

# Context Awareness
- Remember this is a voice conversation - responses should sound natural when spoken
- Acknowledge when you're working: "Let me check that for you" or "I'm looking into that now"
- If something will take time, say so: "This might take a moment"
- Use verbal cues to indicate you're done speaking: "What do you think?" or "How does that sound?"

When the user mentions coding problems, focus on understanding the issue first before jumping to solutions.
Be a helpful, friendly AI coworker who respects the user's time and attention."""


VOICE_FORMAT_PROMPT = """Format your response for voice interaction:

- Main response should be 1-3 sentences that sound natural when spoken aloud
- If you include code, DON'T describe it in detail in voice - just mention you've prepared it
- End with a clear question or statement that invites a response
- Use conversational language, contractions, and natural speech patterns

Example good voice responses:
- "I've found the issue in your DBT model and prepared a fix. Should I apply it?"
- "I see what's happening. The problem is with the join condition. I've written the corrected code for you."
- "That's an interesting challenge! Let me create a solution for you. One moment..."

Example bad voice responses:
- Reading code line by line
- Long technical explanations without breaks
- Formal, robotic language
- Multiple questions at once"""
