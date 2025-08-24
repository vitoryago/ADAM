"""
Message management endpoints for ADAM v2.0
Handles sending messages, LLM integration, and streaming responses
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, AsyncGenerator
import json
import logging
from datetime import datetime

from database import get_db
from models import (
    Message, MessageCreate, MessageResponse,
    Conversation, Project
)
from services.llm_service import LLMService
from services.memory_service import ProjectMemoryService, ADAM_MEMORY_AVAILABLE
from services.tool_service import get_tool_service
from services.workflow_service import get_workflow_service
from services.agent_service import get_agent_service
from tools.file_system_tools import FileSystemTools
from tools.file_operation_handler import FileOperationHandler

router = APIRouter()
logger = logging.getLogger(__name__)
tool_service = get_tool_service()
file_handler = FileOperationHandler()
workflow_service = get_workflow_service()
agent_service = get_agent_service()


@router.post("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def send_message(
    conversation_id: str,
    message_data: MessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Send a message and get AI response"""
    # Verify conversation exists and get project
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get project for settings and memory
    project_result = await db.execute(
        select(Project).where(Project.id == conversation.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # Create user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message_data.content,
        has_image=message_data.has_image,
        image_url=f"data:image/jpeg;base64,{message_data.image_data}" if message_data.has_image and message_data.image_data else None
    )
    
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)
    
    # Get conversation history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(10)  # Last 10 messages for context
    )
    history = history_result.scalars().all()
    
    # Initialize services with project ID for memory storage
    llm_service = LLMService(project_settings=project.settings, project_id=project.id)
    
    # Get memory context if enabled
    memory_context = ""
    if message_data.use_memory and ADAM_MEMORY_AVAILABLE:
        try:
            # Try to use advanced memory service
            try:
                from services.advanced_memory_service import AdvancedMemoryService
                memory_service = AdvancedMemoryService(project.id, project.name)
                
                # Use advanced search if available
                # Don't filter by conversation_id to get memories across all conversations
                memories = await memory_service.advanced_search(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=5,
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=5
                )
            
            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:3]:  # Top 3 memories
                    memory_context += f"- {mem.content[:200]}...\n"
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
    
    # Check if message requires agent/tool execution
    tool_result = None
    tool_output = ""
    
    try:
        # Determine if this needs agent execution
        message_lower = message_data.content.lower()
        needs_agent = any(word in message_lower for word in [
            'check', 'explore', 'navigate', 'find', 'search',
            'create', 'write', 'edit', 'read', 'list',
            'folder', 'directory', 'file', 'model', 'repository', 'banking', 'crypto'
        ])
        
        # Extract workspace path
        workspace_path = None
        if message_data.workspace_context:
            workspace = message_data.workspace_context.get('workspace', {})
            if workspace and not workspace.get('error'):
                folders = workspace.get('folders', [])
                if folders and folders[0].get('path'):
                    workspace_path = folders[0]['path']
        
        if needs_agent:
            logger.info(f"🚀 Autonomous agent execution triggered for: {message_data.content[:100]}...")
            
            # Use the new LangGraph orchestrator for autonomous execution
            try:
                from agents.orchestrator import get_orchestrator
                orchestrator = get_orchestrator()
                
                logger.info("🤖 Running autonomous agent workflow...")
                
                # Process the request autonomously
                result = await orchestrator.process_request(
                    user_message=message_data.content,
                    workspace_path=workspace_path or "/Users/vitoryago"
                )
                
                if result['status'] == 'completed':
                    tool_output = result['response']
                    logger.info(f"✅ Autonomous execution completed in {result['execution_time']:.2f}s")
                    logger.info(f"📊 Executed {result['tasks_executed']} tasks")
                    tool_result = result
                else:
                    logger.error(f"❌ Autonomous execution failed: {result.get('error')}")
                    tool_output = f"I encountered an error: {result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                logger.error(f"❌ Failed to run autonomous agent: {e}")
                # Fallback to old agent service if needed
                if agent_service and agent_service.runtime:
                    agent_service.runtime.workspace_path = workspace_path
                    agent_result = await agent_service.execute_immediate(message_data.content)
                    if agent_result and agent_result.get('status') == 'success':
                        tool_output = f"\n\n{agent_result.get('output', '')}\n"
                        tool_result = agent_result
        
        # Fallback to simple tool service
        if not tool_result:
            tool_result = tool_service.process_tool_request(message_data.content)
            
            if tool_result and tool_result.get('status') == 'success':
                tool_output = f"\n\n**Tool Result:**\n```\n{tool_result.get('output', '')}\n```\n"
                logger.info(f"Tool executed: {tool_result.get('metadata', {})}")
                
    except Exception as e:
        logger.error(f"❌ Error in agent/tool execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Process workspace context and file operations
    workspace_info = ""
    file_operation_results = ""
    
    # Set workspace path for file handler if provided
    if message_data.workspace_context:
        try:
            workspace = message_data.workspace_context.get('workspace', {})
            active_file = message_data.workspace_context.get('activeFile', {})
            
            # Set workspace path for file operations
            if workspace and not workspace.get('error'):
                folders = workspace.get('folders', [])
                if folders and folders[0].get('path'):
                    file_handler.file_tools.workspace_path = folders[0]['path']
            
            # ALWAYS process file operations for ANY query
            # The handler will determine if file operations are needed
            try:
                file_ops_result = file_handler.process_query(message_data.content)
                
                # If file operations were performed, include the results
                if file_ops_result and file_ops_result.get('formatted_output'):
                    file_operation_results = file_ops_result['formatted_output']
                    logger.info(f"File operations performed: {file_ops_result.get('operations_performed', [])}")
            except Exception as e:
                logger.error(f"Error processing file operations: {e}")
                # Continue without file operations rather than failing the whole request
            
            # Add workspace context to the message
            if not active_file.get('error') and active_file.get('content'):
                workspace_info = f"\n\n[Current File: {active_file.get('file', 'unknown')} ({active_file.get('language', 'unknown')})]"
                workspace_info += f"\n```{active_file.get('language', '')}\n{active_file.get('content', '')[:5000]}\n```"
            
            if workspace and not workspace.get('error'):
                folders = workspace.get('folders', [])
                if folders:
                    workspace_info += f"\n[Workspace: {folders[0].get('name', 'Unknown')} at {folders[0].get('path', 'Unknown')}]"
        except Exception as e:
            logger.error(f"Error processing workspace context: {e}")
    
    # Generate AI response (incorporating tool results and workspace context)
    try:
        # Check if we already have a complete response from the orchestrator
        if tool_result and tool_result.get('status') == 'completed' and tool_result.get('response'):
            # Use the orchestrator's response directly
            response_content = tool_result['response']
            # Create a mock response object for compatibility
            from types import SimpleNamespace
            response = SimpleNamespace(
                content=response_content,
                model_used=message_data.model or "gpt-5",
                tokens_used=0,  # Would need to track this in orchestrator
                cost=0.0,  # Would need to track this in orchestrator
                metadata={}
            )
        else:
            # Generate response with LLM (for non-orchestrator queries)
            # Modify the message to include tool results, file operations and workspace context
            enhanced_message = message_data.content
            if tool_output:
                enhanced_message += tool_output
            if file_operation_results:
                enhanced_message += file_operation_results
            if workspace_info:
                enhanced_message += workspace_info
            
            # Build enhanced system prompt if workspace context exists
            system_prompt = message_data.system_prompt
            if message_data.workspace_context and not system_prompt:
                system_prompt = """You are ADAM, an AI assistant with access to the user's VSCode workspace. 
You can help with code analysis, understanding, and providing guidance.
File operation results will appear in your context automatically when you mention files."""
            
            response = await llm_service.generate_response(
                message=enhanced_message,
                history=history,
                memory_context=memory_context,
                model=message_data.model,
                system_prompt=system_prompt,
                image_data=message_data.image_data if message_data.has_image else None,
                use_search=message_data.use_search,
                search_mode=message_data.search_mode
            )
            
            # Ensure response is not empty
            response_content = response.content if response.content else "I understand your request. Let me help you with that."
        
        # Create assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_content,
            model=response.model_used,
            tokens_used=response.tokens_used,
            cost=response.cost
        )
        
        # If response includes citations, store them in metadata (would need schema update)
        # For now, we'll include citations in the response content if available
        if response.metadata and 'citations' in response.metadata:
            citations = response.metadata['citations']
            if citations:
                citation_text = "\n\n---\n**Sources:**\n"
                for i, citation in enumerate(citations, 1):
                    citation_text += f"{i}. [{citation.get('title', 'Source')}]({citation.get('url', '#')})\n"
                assistant_message.content += citation_text
        
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)
        
        # Store in memory if worthy
        # Lower threshold for valuable content like code
        is_code_content = "```" in response.content or "def " in response.content or "import " in response.content
        # Check for substantial content (DBT, models, etc.)
        is_substantial = len(response.content.split()) > 100 or any(term in response.content.upper() for term in ["DBT", "PDT", "MODEL", "SQL", "CREATE", "SELECT"])
        
        # Much lower thresholds to ensure memory storage
        cost_threshold = 0.00001 if (is_code_content or is_substantial) else 0.0001
        token_threshold = 200 if is_substantial else 500
        
        logger.info(f"Memory check - use_memory: {message_data.use_memory}, cost: {response.cost}, threshold: {cost_threshold}, tokens: {response.tokens_used}, substantial: {is_substantial}")
        
        if message_data.use_memory and (response.cost > cost_threshold or response.tokens_used > token_threshold):
            try:
                memory_service = ProjectMemoryService(project.id, project.name)
                await memory_service.store_memory(
                    content=f"Q: {message_data.content}\n\nA: {response.content}",
                    memory_type="conversation",
                    metadata={
                        "model": response.model_used,
                        "cost": response.cost,
                        "tokens": response.tokens_used
                    },
                    conversation_id=conversation_id,
                    cost=response.cost
                )
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
        
        # Return both messages
        return [
            MessageResponse.model_validate(user_message),
            MessageResponse.model_validate(assistant_message)
        ]
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    message_data: MessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Send a message and stream the AI response"""
    # Verify conversation and get project (same as above)
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get project
    project_result = await db.execute(
        select(Project).where(Project.id == conversation.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # Create user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=message_data.content,
        has_image=message_data.has_image,
        image_url=f"data:image/jpeg;base64,{message_data.image_data}" if message_data.has_image and message_data.image_data else None
    )
    
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)
    
    # Get history
    history_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(10)
    )
    history = history_result.scalars().all()
    
    # Initialize services with project ID for memory storage
    llm_service = LLMService(project_settings=project.settings, project_id=project.id)
    
    # Get memory context
    memory_context = ""
    if message_data.use_memory and ADAM_MEMORY_AVAILABLE:
        try:
            # Try to use advanced memory service
            try:
                from services.advanced_memory_service import AdvancedMemoryService
                memory_service = AdvancedMemoryService(project.id, project.name)
                
                # Use advanced search if available
                # Don't filter by conversation_id to get memories across all conversations
                memories = await memory_service.advanced_search(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=5,
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=5
                )
            
            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:3]:
                    memory_context += f"- {mem.content[:200]}...\n"
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
    
    async def stream_response():
        """Stream the response as Server-Sent Events"""
        try:
            # Stream user message first
            yield f"data: {json.dumps({'type': 'user_message', 'id': user_message.id, 'content': user_message.content})}\n\n"
            
            # Stream AI response
            full_response = ""
            tokens_used = 0
            cost = 0.0
            model_used = message_data.model or "automatic"
            
            async for chunk in llm_service.stream_response(
                message=message_data.content,
                history=history,
                memory_context=memory_context,
                model=message_data.model,
                image_data=message_data.image_data if message_data.has_image else None,
                use_search=message_data.use_search,
                search_mode=message_data.search_mode
            ):
                full_response += chunk.content
                tokens_used = chunk.tokens_used
                cost = chunk.cost
                model_used = chunk.model_used
                
                # Send chunk
                yield f"data: {json.dumps({'type': 'assistant_chunk', 'content': chunk.content})}\n\n"
            
            # Create assistant message in database
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                model=model_used,
                tokens_used=tokens_used,
                cost=cost
            )
            
            db.add(assistant_message)
            await db.commit()
            await db.refresh(assistant_message)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'id': assistant_message.id, 'tokens': tokens_used, 'cost': cost, 'model': model_used})}\n\n"
            
            # Store in memory if worthy
            # Lower threshold for valuable content like code
            is_code_content = "```" in full_response or "def " in full_response or "import " in full_response
            # Check for substantial content (DBT, models, etc.)
            is_substantial = len(full_response.split()) > 100 or any(term in full_response.upper() for term in ["DBT", "PDT", "MODEL", "SQL", "CREATE", "SELECT"])
            
            # Much lower thresholds to ensure memory storage
            cost_threshold = 0.00001 if (is_code_content or is_substantial) else 0.0001
            token_threshold = 200 if is_substantial else 500
            
            logger.info(f"Memory check (streaming) - use_memory: {message_data.use_memory}, cost: {cost}, threshold: {cost_threshold}, tokens: {tokens_used}, substantial: {is_substantial}")
            
            if message_data.use_memory and (cost > cost_threshold or tokens_used > token_threshold):
                try:
                    memory_service = ProjectMemoryService(project.id, project.name)
                    await memory_service.store_memory(
                        content=f"Q: {message_data.content}\n\nA: {full_response}",
                        memory_type="conversation",
                        metadata={
                            "model": model_used,
                            "cost": cost,
                            "tokens": tokens_used
                        },
                        conversation_id=conversation_id,
                        cost=cost
                    )
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")
                    
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get messages in a conversation"""
    # Verify conversation exists
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()
    
    # Reverse to show oldest first
    messages.reverse()
    
    return [MessageResponse.model_validate(msg) for msg in messages]


@router.delete("/messages/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a message (soft delete for audit trail)"""
    result = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    message = result.scalar_one_or_none()
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    # For now, we'll actually delete. In production, consider soft delete
    await db.delete(message)
    await db.commit()


@router.post("/conversations/{conversation_id}/regenerate")
async def regenerate_last_response(
    conversation_id: str,
    model: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Regenerate the last assistant response"""
    # Get last two messages (should be user then assistant)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(2)
    )
    last_messages = result.scalars().all()
    
    if len(last_messages) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough messages to regenerate"
        )
    
    # Verify pattern is assistant then user (reversed due to desc order)
    if last_messages[0].role != "assistant" or last_messages[1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from assistant to regenerate"
        )
    
    # Delete the last assistant message
    await db.delete(last_messages[0])
    await db.commit()
    
    # Resend the user message
    message_data = MessageCreate(
        content=last_messages[1].content,
        model=model,
        has_image=last_messages[1].has_image,
        use_memory=True
    )
    
    return await send_message(conversation_id, message_data, db)