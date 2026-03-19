"""
Markdown rendering service for ADAM
"""
import markdown
from markdown.extensions import fenced_code, tables, nl2br
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter

class MarkdownService:
    def __init__(self):
        self.md = markdown.Markdown(extensions=[
            'fenced_code',
            'tables',
            'nl2br',
            'codehilite',
            'toc',
            'attr_list'
        ])

    def render_to_html(self, content: str) -> str:
        """Convert markdown to HTML with syntax highlighting"""
        # Reset the Markdown instance
        self.md.reset()

        # Convert markdown to HTML
        html = self.md.convert(content)

        # Add dark theme classes
        html = html.replace('<pre>', '<pre class="bg-gray-900 p-4 rounded-lg overflow-x-auto">')
        html = html.replace('<code>', '<code class="bg-gray-800 px-1 py-0.5 rounded text-sm">')

        return html

    def extract_code_blocks(self, content: str) -> list:
        """Extract code blocks from markdown content"""
        import re
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)

        return [{'language': lang or 'text', 'code': code} for lang, code in matches]

# Singleton instance
markdown_service = MarkdownService()
