"""
UI routes for HTMX interactions
"""
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import logging
from html import escape
import json

from database import get_db
from models import Project, Conversation, Message
from services.markdown_service import markdown_renderer

router = APIRouter()
logger = logging.getLogger(__name__)


def render_markdown_server(text: str) -> str:
    """Render markdown to HTML on the server side"""
    if not text:
        return ''
    
    # Escape HTML first to prevent XSS
    text = escape(text)
    
    # Code blocks with language
    def replace_code_block(match):
        lang = match.group(1) or 'plaintext'
        code = match.group(2).strip()
        
        # Language display names
        lang_display = {
            'js': 'JavaScript', 'javascript': 'JavaScript',
            'py': 'Python', 'python': 'Python',
            'bash': 'Bash', 'sh': 'Shell', 'shell': 'Shell',
            'sql': 'SQL', 'json': 'JSON',
            'yaml': 'YAML', 'yml': 'YAML',
            'html': 'HTML', 'css': 'CSS',
            'typescript': 'TypeScript', 'ts': 'TypeScript',
            'go': 'Go', 'rust': 'Rust', 'java': 'Java',
            'cpp': 'C++', 'c': 'C', 'cs': 'C#',
            'php': 'PHP', 'ruby': 'Ruby', 'r': 'R',
            'swift': 'Swift', 'kotlin': 'Kotlin',
            'scala': 'Scala', 'dockerfile': 'Dockerfile',
            'xml': 'XML', 'markdown': 'Markdown', 'md': 'Markdown'
        }
        
        display_name = lang_display.get(lang.lower(), lang.upper())
        
        return f'''<div class="relative group my-4">
            <div class="bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
                <div class="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center justify-between">
                    <span class="text-xs text-gray-400 font-medium">{display_name}</span>
                    <button onclick="copyCode(this)" class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity">Copy</button>
                </div>
                <pre class="p-4 overflow-x-auto"><code class="language-{lang} text-sm font-mono">{code}</code></pre>
            </div>
        </div>'''
    
    # Handle code blocks with optional language and newlines/spaces
    text = re.sub(r'```(\w*)\s*([\s\S]*?)```', replace_code_block, text)
    
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code class="bg-gray-800 px-1 py-0.5 rounded text-sm font-mono">\1</code>', text)
    
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3 class="text-lg font-semibold mt-4 mb-2">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2 class="text-xl font-semibold mt-4 mb-2">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1 class="text-2xl font-bold mt-4 mb-2">\1</h1>', text, flags=re.MULTILINE)
    
    # Bold and italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Lists
    text = re.sub(r'^(\s*)[-*+] (.+)$', r'\1• \2', text, flags=re.MULTILINE)
    text = re.sub(r'^(\s*)(\d+)\. (.+)$', r'\1\2. \3', text, flags=re.MULTILINE)
    
    # Line breaks
    text = text.replace('\n', '<br>')
    
    return text


@router.get("/projects", response_class=HTMLResponse)
async def projects_list(request: Request, db: AsyncSession = Depends(get_db)):
    """Render project cards for HTMX"""
    # Get all projects with stats
    result = await db.execute(
        select(Project)
        .where(Project.is_archived == False)
        .order_by(Project.updated_at.desc())
    )
    projects = result.scalars().all()
    
    # Calculate stats for each project
    project_data = []
    for project in projects:
        # Get conversation count
        conv_count = await db.execute(
            select(func.count(Conversation.id))
            .where(Conversation.project_id == project.id)
        )
        conversation_count = conv_count.scalar() or 0
        
        # Get memory stats from the service if available
        try:
            from services.memory_service import ProjectMemoryService
            memory_service = ProjectMemoryService(project.id, project.name)
            memory_stats = await memory_service.get_memory_stats()
            memory_count = memory_stats.get('total_memories', 0)
        except:
            memory_count = 0
        
        project_data.append({
            'project': project,
            'conversation_count': conversation_count,
            'memory_count': memory_count,
            'total_cost': 0.0  # TODO: Calculate from messages
        })
    
    html = '<div id="project-cards">'
    for data in project_data:
        project = data['project']
        html += f'''
        <div class="bg-gray-800 rounded-lg p-6 hover:bg-gray-750 transition-colors cursor-pointer" 
             onclick="window.location.href='/project/{project.id}'">
            <div class="flex justify-between items-start mb-4">
                <h3 class="text-xl font-semibold">{project.name}</h3>
                <div class="flex space-x-2">
                    <button 
                        hx-get="/projects/{project.id}/edit"
                        hx-target="#modal-container"
                        hx-swap="innerHTML"
                        onclick="event.stopPropagation()"
                        class="p-1 hover:bg-gray-700 rounded"
                    >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path>
                        </svg>
                    </button>
                    <button 
                        hx-delete="/api/projects/{project.id}"
                        hx-confirm="Are you sure you want to delete this project?"
                        onclick="event.stopPropagation()"
                        class="p-1 hover:bg-gray-700 rounded text-red-400"
                    >
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                        </svg>
                    </button>
                </div>
            </div>
            
            <p class="text-gray-400 text-sm mb-4">{project.description or "No description"}</p>
            
            <div class="grid grid-cols-3 gap-4 text-sm">
                <div>
                    <div class="text-gray-500">Conversations</div>
                    <div class="font-semibold">{data['conversation_count']}</div>
                </div>
                <div>
                    <div class="text-gray-500">Memories</div>
                    <div class="font-semibold">{data['memory_count']}</div>
                </div>
                <div>
                    <div class="text-gray-500">Cost</div>
                    <div class="font-semibold">${data['total_cost']:.2f}</div>
                </div>
            </div>
            
            <div class="mt-4 pt-4 border-t border-gray-700">
                <div class="text-xs text-gray-500">
                    Created {project.created_at.strftime('%B %d, %Y')}
                </div>
            </div>
        </div>
        '''
    
    if not project_data:
        html += '''
        <div class="col-span-full text-center py-12">
            <svg class="w-16 h-16 mx-auto text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"></path>
            </svg>
            <p class="text-gray-500">No projects yet. Create your first project to get started!</p>
        </div>
        '''
    
    html += '</div>'
    return HTMLResponse(content=html)


@router.get("/projects/new", response_class=HTMLResponse)
async def new_project_modal(request: Request):
    """Render new project modal"""
    html = '''
    <div class="fixed inset-0 z-50 overflow-y-auto" x-data="{ open: true }" x-show="open">
        <div class="flex items-center justify-center min-h-screen px-4">
            <div class="fixed inset-0 bg-black opacity-50" @click="open = false; document.getElementById('modal-container').innerHTML = ''"></div>
            
            <div class="relative bg-gray-800 rounded-lg max-w-md w-full p-6" @click.away="open = false; document.getElementById('modal-container').innerHTML = ''">
                <h2 class="text-xl font-semibold mb-4">Create New Project</h2>
                
                <form hx-post="/api/projects/" 
                      hx-ext="json-enc"
                      hx-swap="none" 
                      hx-on::after-request="if(event.detail.successful) { window.location.href = '/project/' + JSON.parse(event.detail.xhr.responseText).id; }"
                      hx-on::response-error="showToast('Failed to create project: ' + event.detail.xhr.responseText, 'error')">
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Project Name</label>
                        <input 
                            type="text" 
                            name="name" 
                            required
                            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="My AI Project"
                        >
                    </div>
                    
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Description (Optional)</label>
                        <textarea 
                            name="description" 
                            rows="3"
                            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="What is this project about?"
                        ></textarea>
                    </div>
                    
                    <div class="mb-6">
                        <label class="block text-sm font-medium mb-2">Default Model</label>
                        <select 
                            name="settings[model]"
                            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="">Automatic (Smart Routing)</option>
                            <option value="grok-3-mini-high">Grok-3 Mini (Fast)</option>
                            <option value="grok-4">Grok-4 (Balanced)</option>
                            <option value="grok-4-reasoning">Grok-4 Reasoning (Complex)</option>
                        </select>
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button 
                            type="button"
                            @click="open = false; document.getElementById('modal-container').innerHTML = ''"
                            class="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                        >
                            Cancel
                        </button>
                        <button 
                            type="submit"
                            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                        >
                            Create Project
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    '''
    return HTMLResponse(content=html)


@router.get("/conversations/new/{project_id}", response_class=HTMLResponse)
async def new_conversation_modal(project_id: str, request: Request):
    """Render new conversation modal"""
    html = f'''
    <div class="fixed inset-0 z-50 overflow-y-auto" x-data="{{ open: true }}" x-show="open">
        <div class="flex items-center justify-center min-h-screen px-4">
            <div class="fixed inset-0 bg-black opacity-50" @click="open = false; document.getElementById('modal-container').innerHTML = ''"></div>
            
            <div class="relative bg-gray-800 rounded-lg max-w-md w-full p-6" @click.away="open = false; document.getElementById('modal-container').innerHTML = ''">
                <h2 class="text-xl font-semibold mb-4">New Conversation</h2>
                
                <form hx-post="/api/projects/{project_id}/conversations" 
                      hx-ext="json-enc"
                      hx-swap="none" 
                      hx-on::after-request="if(event.detail.successful) {{ location.reload(); }}"
                      hx-on::response-error="showToast('Failed to create conversation', 'error')">
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Title</label>
                        <input 
                            type="text" 
                            name="title" 
                            required
                            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="General Discussion"
                        >
                    </div>
                    
                    <div class="flex justify-end space-x-3">
                        <button 
                            type="button"
                            @click="open = false; document.getElementById('modal-container').innerHTML = ''"
                            class="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                        >
                            Cancel
                        </button>
                        <button 
                            type="submit"
                            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                        >
                            Create
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    '''
    return HTMLResponse(content=html)


@router.get("/api/projects/stats")
async def project_stats(db: AsyncSession = Depends(get_db)):
    """Get overall project statistics"""
    # Count projects
    project_count = await db.execute(
        select(func.count(Project.id)).where(Project.is_archived == False)
    )
    total_projects = project_count.scalar() or 0
    
    # Count active conversations
    conv_count = await db.execute(
        select(func.count(Conversation.id))
    )
    active_conversations = conv_count.scalar() or 0
    
    # Calculate total cost from messages
    cost_result = await db.execute(
        select(func.sum(Message.cost)).where(Message.cost != None)
    )
    total_cost = cost_result.scalar() or 0.0
    
    # Get total memories (aggregate from all projects)
    total_memories = 0
    try:
        from services.memory_service import ProjectMemoryService
        projects_result = await db.execute(select(Project))
        projects = projects_result.scalars().all()
        
        for project in projects:
            try:
                memory_service = ProjectMemoryService(project.id, project.name)
                stats = await memory_service.get_memory_stats()
                total_memories += stats.get('total_memories', 0)
            except:
                continue
    except:
        pass
    
    return {
        "total_projects": total_projects,
        "active_conversations": active_conversations,
        "total_memories": total_memories,
        "total_cost": total_cost
    }


@router.get("/conversations/{conversation_id}/messages/html", response_class=HTMLResponse)
async def get_messages_html(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """Get messages as HTML for HTMX"""
    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    
    html = '<div class="space-y-4">'
    
    for msg in messages:
        if msg.role == "user":
            alignment = "justify-end"
            bg_color = "bg-blue-600 text-white"
            prose_style = "prose-invert"
        else:
            alignment = ""
            bg_color = "bg-gray-800"
            prose_style = ""
        
        # Render markdown server-side
        if msg.role == "user":
            # For user messages, just escape HTML
            rendered_content = escape(msg.content).replace('\n', '<br>')
        else:
            # For assistant messages, render markdown
            rendered_content = markdown_renderer.render(msg.content)
        
        html += f'''
        <div class="flex {alignment} message-enter">
            <div class="max-w-3xl {bg_color} rounded-lg px-4 py-3">
                <div class="prose prose-sm {prose_style} max-w-none">
                    {rendered_content}
                </div>
                '''
        
        # Add metadata for assistant messages
        if msg.role == "assistant" and msg.model:
            html += f'''
                <div class="mt-2 text-xs text-gray-400">
                    {msg.model} · {msg.tokens_used or 0} tokens · ${msg.cost or 0:.4f}
                </div>
            '''
        
        html += '''
            </div>
        </div>
        '''
    
    if not messages:
        html += '''
        <div class="text-center text-gray-500 py-8">
            <p>No messages yet. Start the conversation!</p>
        </div>
        '''
    
    html += '</div>'
    
    return HTMLResponse(content=html)


@router.get("/memories/browse/{project_id}", response_class=HTMLResponse)
async def browse_memories_modal(project_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """Render memory browsing modal"""
    # Get project
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    
    if not project:
        return HTMLResponse(content="Project not found", status_code=404)
    
    html = f'''
    <div class="fixed inset-0 z-50 overflow-y-auto" x-data="{{ open: true, searchQuery: '' }}" x-show="open">
        <div class="flex items-center justify-center min-h-screen px-4">
            <div class="fixed inset-0 bg-black opacity-50" @click="open = false; document.getElementById('modal-container').innerHTML = ''"></div>
            
            <div class="relative bg-gray-800 rounded-lg max-w-4xl w-full max-h-[80vh] flex flex-col" @click.away="open = false; document.getElementById('modal-container').innerHTML = ''">
                <div class="p-6 border-b border-gray-700">
                    <h2 class="text-xl font-semibold mb-4">Project Memories: {project.name}</h2>
                    
                    <div class="relative">
                        <input 
                            type="text"
                            x-model="searchQuery"
                            @keyup.enter="htmx.ajax('POST', '/api/projects/{project_id}/memories/search', {{target: '#memory-results', values: {{query: searchQuery, limit: 20}}}})"
                            class="w-full bg-gray-700 border border-gray-600 rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="Search memories..."
                        >
                        <svg class="absolute left-3 top-2.5 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                    </div>
                </div>
                
                <div class="flex-1 overflow-y-auto p-6">
                    <div id="memory-results" class="space-y-4">
                        <!-- Memory stats -->
                        <div 
                            hx-get="/api/projects/{project_id}/memories/stats"
                            hx-trigger="load"
                            hx-swap="innerHTML"
                            class="bg-gray-700 rounded-lg p-4"
                        >
                            <div class="animate-pulse">
                                <div class="h-4 bg-gray-600 rounded w-1/2 mb-2"></div>
                                <div class="h-4 bg-gray-600 rounded w-3/4"></div>
                            </div>
                        </div>
                        
                        <div class="text-center text-gray-500 py-8">
                            <p>Enter a search query to browse memories</p>
                        </div>
                    </div>
                </div>
                
                <div class="p-4 border-t border-gray-700 flex justify-between">
                    <button 
                        hx-get="/api/projects/{project_id}/memories/export"
                        class="px-4 py-2 text-blue-400 hover:text-blue-300 transition-colors"
                    >
                        Export Memories
                    </button>
                    <button 
                        @click="open = false; document.getElementById('modal-container').innerHTML = ''"
                        class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    </div>
    '''
    return HTMLResponse(content=html)