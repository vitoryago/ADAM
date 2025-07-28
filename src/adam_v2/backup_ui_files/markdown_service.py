"""
Markdown rendering service for ADAM v2.0
Provides server-side markdown to HTML conversion with syntax highlighting
"""

import markdown2
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
import re


class MarkdownRenderer:
    """Renders markdown to HTML with syntax highlighting"""
    
    def __init__(self):
        # Configure markdown2 with extras
        self.extras = [
            'fenced-code-blocks',
            'tables',
            'break-on-newline',
            'code-friendly',
            'cuddled-lists',
            'header-ids',
            'spoiler',
            'strike',
            'task_list'
        ]
        
        # Pygments formatter for syntax highlighting
        self.formatter = HtmlFormatter(
            style='monokai',
            cssclass='highlight',
            prestyles='background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; overflow-x: auto;'
        )
    
    def render(self, text: str) -> str:
        """Render markdown to HTML with syntax highlighting"""
        if not text:
            return ''
        
        # First pass: Extract code blocks and replace with placeholders
        code_blocks = []
        code_pattern = r'```(\w*)\n([\s\S]*?)```'
        
        def extract_code(match):
            lang = match.group(1) or 'text'
            code = match.group(2).strip()
            
            # Store the code block
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append((lang, code))
            return placeholder
        
        # Extract code blocks
        text_with_placeholders = re.sub(code_pattern, extract_code, text)
        
        # Render markdown (without code blocks)
        html = markdown2.markdown(text_with_placeholders, extras=self.extras)
        
        # Now render code blocks with syntax highlighting
        for i, (lang, code) in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            
            # Get language display name
            lang_display = {
                'py': 'Python', 'python': 'Python',
                'js': 'JavaScript', 'javascript': 'JavaScript',
                'ts': 'TypeScript', 'typescript': 'TypeScript',
                'sql': 'SQL', 'bash': 'Bash', 'sh': 'Shell',
                'yaml': 'YAML', 'yml': 'YAML',
                'json': 'JSON', 'xml': 'XML', 'html': 'HTML',
                'css': 'CSS', 'go': 'Go', 'rust': 'Rust',
                'java': 'Java', 'cpp': 'C++', 'c': 'C',
                'cs': 'C#', 'php': 'PHP', 'ruby': 'Ruby',
                'r': 'R', 'swift': 'Swift', 'kotlin': 'Kotlin',
                'dockerfile': 'Dockerfile'
            }.get(lang.lower(), lang.upper() if lang else 'Code')
            
            # Try to get lexer for syntax highlighting
            try:
                if lang:
                    lexer = get_lexer_by_name(lang.lower())
                else:
                    lexer = guess_lexer(code)
            except:
                # Fallback to plain text
                from pygments.lexers import TextLexer
                lexer = TextLexer()
            
            # Highlight the code
            highlighted_code = highlight(code, lexer, self.formatter)
            
            # Create the code block HTML
            code_html = f'''
            <div class="code-block-wrapper my-4">
                <div class="code-header bg-gray-800 px-4 py-2 rounded-t-lg flex justify-between items-center">
                    <span class="text-xs text-gray-400 font-medium">{lang_display}</span>
                    <button onclick="copyCode(this)" class="copy-btn text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
                        Copy
                    </button>
                </div>
                <div class="code-content rounded-b-lg overflow-hidden">
                    {highlighted_code}
                </div>
            </div>
            '''
            
            # Replace placeholder with rendered code
            html = html.replace(f"<p>{placeholder}</p>", code_html)
            html = html.replace(placeholder, code_html)
        
        # Add custom styles for other elements
        html = self._style_html(html)
        
        return html
    
    def _style_html(self, html: str) -> str:
        """Add Tailwind classes to HTML elements"""
        # Headers
        html = re.sub(r'<h1>', '<h1 class="text-2xl font-bold mb-4 mt-6">', html)
        html = re.sub(r'<h2>', '<h2 class="text-xl font-semibold mb-3 mt-5">', html)
        html = re.sub(r'<h3>', '<h3 class="text-lg font-semibold mb-2 mt-4">', html)
        
        # Paragraphs
        html = re.sub(r'<p>', '<p class="mb-4">', html)
        
        # Lists
        html = re.sub(r'<ul>', '<ul class="list-disc list-inside mb-4 ml-4">', html)
        html = re.sub(r'<ol>', '<ol class="list-decimal list-inside mb-4 ml-4">', html)
        html = re.sub(r'<li>', '<li class="mb-1">', html)
        
        # Inline code
        html = re.sub(r'<code>([^<]+)</code>', 
                     r'<code class="bg-gray-800 px-1 py-0.5 rounded text-sm font-mono">\1</code>', 
                     html)
        
        # Blockquotes
        html = re.sub(r'<blockquote>', 
                     '<blockquote class="border-l-4 border-gray-600 pl-4 italic my-4">', 
                     html)
        
        # Tables
        html = re.sub(r'<table>', 
                     '<table class="border-collapse table-auto w-full mb-4">', 
                     html)
        html = re.sub(r'<th>', 
                     '<th class="border border-gray-600 px-4 py-2 bg-gray-800">', 
                     html)
        html = re.sub(r'<td>', 
                     '<td class="border border-gray-700 px-4 py-2">', 
                     html)
        
        return html
    
    def get_styles(self) -> str:
        """Get CSS styles for syntax highlighting"""
        return f"""
        <style>
        {self.formatter.get_style_defs('.highlight')}
        .code-block-wrapper {{
            border: 1px solid #374151;
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        .copy-btn {{
            user-select: none;
        }}
        .copy-btn:active {{
            transform: scale(0.95);
        }}
        </style>
        """


# Global renderer instance
markdown_renderer = MarkdownRenderer()