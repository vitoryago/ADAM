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
from services.onboarding_integration_service import OnboardingIntegrationService

router = APIRouter()
logger = logging.getLogger(__name__)


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
        .limit(30)  # Last 30 messages for better context retention
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
                    limit=15,  # Increased from 5 to retrieve more relevant memories
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=20  # Further increased to retrieve more relevant memories for better context
                )

            if memories:
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:10]:  # Use up to 10 memories in context (was 3)  # Top 3 memories
                    memory_context += f"- {mem.content[:200]}...\n"
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")

    # Check for DBT questions and add context
    dbt_context = ""
    workspace_path = None
    if message_data.workspace_context and message_data.workspace_context.get('workspaceFolder'):
        workspace_path = message_data.workspace_context['workspaceFolder']

    if workspace_path:
        try:
            from services.dbt_integration_service import get_dbt_integration_service
            dbt_service = get_dbt_integration_service()

            # Check if this is a DBT question
            if dbt_service.is_dbt_question(message_data.content, workspace_path):
                dbt_context = dbt_service.get_dbt_context(message_data.content, workspace_path)
                logger.info(f"Added DBT context for question: {message_data.content[:50]}")
        except Exception as e:
            logger.warning(f"Error adding DBT context: {e}")
    
    # Check if VSCode sent workspace context with file content
    enhanced_content = message_data.content
    workspace_instructions = ""
    
    # Check for file operation requests AND set concise mode for VSCode
    if message_data.workspace_context:
        # Always use concise style in VSCode unless specified
        if not message_data.response_style:
            message_data.response_style = "concise"
        
        # Use Claude Haiku to intelligently detect file operations needed
        try:
            from services.fast_routing_service import FastRoutingService
            routing_service = FastRoutingService()
            
            # Ask Haiku what operations are needed
            operation_prompt = f"""Analyze this VSCode user request and determine what file operations are needed.
User request: "{message_data.content}"
Active file: {message_data.workspace_context.get('activeFile', {}).get('file', 'none')}

Respond with ONLY valid JSON:
{{"needs_file_creation": true/false, "needs_file_edit": true/false, "action_type": "create|edit|debug|analyze|none"}}"""
            
            operation_response = await routing_service.client.complete(
                prompt=operation_prompt,
                model="claude-3.5-haiku",
                temperature=0.0,
                max_tokens=50,
                stream=False
            )
            
            import json
            operation_info = json.loads(operation_response.content.strip())
            
            if operation_info.get("needs_file_creation"):
                workspace_instructions = "\n\n[SYSTEM: User is in VSCode. You can create files by responding with: <<<CREATE_FILE:path/to/file.ext>>> followed by the file content and <<<END_FILE>>>. Be concise.]"
            elif operation_info.get("needs_file_edit") or operation_info.get("action_type") in ["debug", "edit", "fix"]:
                workspace_instructions = "\n\n[SYSTEM: User is in VSCode. When fixing/debugging code, ALWAYS provide the corrected version using: <<<CREATE_FILE:path/to/file.ext>>> with the FULL corrected content and <<<END_FILE>>>. Don't just explain - fix it! Be concise.]"
            
        except Exception as e:
            logger.warning(f"Failed to use Haiku for operation detection: {e}, using fallback")
            # Minimal fallback - just check for obvious keywords
            if "create" in message_data.content.lower() and "file" in message_data.content.lower():
                workspace_instructions = "\n\n[SYSTEM: User is in VSCode. You can create files. Be concise.]"
    
    if message_data.workspace_context and message_data.workspace_context.get('activeFile'):
        active_file = message_data.workspace_context['activeFile']
        if isinstance(active_file, dict) and active_file.get('content'):
            # Detect if the user is referring to a file/code contextually
            content_lower = message_data.content.lower()
            
            # Keywords that suggest the user is referring to something contextual
            file_reference_indicators = ['this', 'that', 'here', 'current', 'above', 'it', 'the file', 'the code']
            action_verbs = ['read', 'analyze', 'explain', 'review', 'check', 'look', 'help', 'understand', 'optimize', 'improve', 'debug', 'fix']
            
            # Check if the message seems to reference the active file
            seems_to_reference_file = (
                # Short questions like "What about this?" or "Can you read it?"
                (len(content_lower.split()) <= 8 and any(ref in content_lower for ref in file_reference_indicators)) or
                # Any action verb with a reference indicator
                (any(verb in content_lower for verb in action_verbs) and any(ref in content_lower for ref in file_reference_indicators)) or
                # Very short contextual questions
                content_lower in ['?', 'and this?', 'this one?', 'how about this?', 'what about this one?']
            )
            
            if seems_to_reference_file:
                file_info = f"\n\n**Current file: {active_file.get('file', 'unknown')}** ({active_file.get('language', 'unknown')} file)\n"
                file_info += f"```{active_file.get('language', '')}\n{active_file.get('content', '')}\n```"
                enhanced_content = f"{message_data.content}\n{file_info}"
    
    # Add workspace instructions if any
    enhanced_content = enhanced_content + workspace_instructions
    
    # Check for onboarding request
    onboarding_service = OnboardingIntegrationService(llm_service)
    if onboarding_service.is_onboarding_request(message_data.content):
        try:
            # Process as onboarding request
            onboarding_response = await onboarding_service.process_onboarding_request(
                message=message_data.content,
                project_id=project.id,
                project_name=project.name,
                project_path=project.settings.get("project_path")
            )
            
            if onboarding_response["type"] == "onboarding":
                # Create assistant message with onboarding content
                assistant_message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=onboarding_response["content"],
                    metadata=onboarding_response.get("metadata", {})
                )
                
                db.add(assistant_message)
                await db.commit()
                await db.refresh(assistant_message)
                
                # Return both messages
                return [
                    MessageResponse.model_validate(user_message),
                    MessageResponse.model_validate(assistant_message)
                ]
        except Exception as e:
            logger.error(f"Error in onboarding processing: {e}")
            # Fall through to regular message handling
    
    # Combine memory and DBT context
    combined_context = memory_context + dbt_context

    # Generate AI response
    try:
        response = await llm_service.generate_response(
            message=enhanced_content,
            history=history,
            memory_context=combined_context,
            model=message_data.model,
            image_data=message_data.image_data if message_data.has_image else None,
            use_search=message_data.use_search,
            search_mode=message_data.search_mode,
            response_style=message_data.response_style
        )
        
        # Create assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.content,
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
        cost_threshold = 0.000001  # Ultra-low threshold to store virtually all responses
        token_threshold = 200 if is_substantial else 500
        
        logger.info(f"Memory check - use_memory: {message_data.use_memory}, cost: {response.cost}, threshold: {cost_threshold}, tokens: {response.tokens_used}, substantial: {is_substantial}")
        
        if message_data.use_memory and (response.cost > cost_threshold or response.tokens_used > token_threshold):
            try:
                memory_service = ProjectMemoryService(project.id, project.name)
                memory_id = await memory_service.store_memory(
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
                logger.info(f"Stored memory {memory_id} for conversation {conversation_id}, cost: {response.cost}, tokens: {response.tokens_used}")
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
    
    # Check if VSCode sent workspace context with file content  
    enhanced_content = message_data.content
    workspace_instructions = ""
    
    # Set concise mode for VSCode and detect file operations
    if message_data.workspace_context:
        # Always use concise style in VSCode unless specified
        if not message_data.response_style:
            message_data.response_style = "concise"
        
        # Detect file operations
        if any(phrase in message_data.content.lower() for phrase in ['debug', 'fix', 'create a file', 'edit this', 'modify this']):
            workspace_instructions = "\n\n[SYSTEM: User is in VSCode. For code fixes, provide the corrected version using <<<CREATE_FILE:filename>>> with full content and <<<END_FILE>>>. Be concise.]"
    
    if message_data.workspace_context and message_data.workspace_context.get('activeFile'):
        active_file = message_data.workspace_context['activeFile']
        if isinstance(active_file, dict) and active_file.get('content'):
            # Detect if the user is referring to a file/code contextually
            content_lower = message_data.content.lower()
            
            # Keywords that suggest the user is referring to something contextual
            file_reference_indicators = ['this', 'that', 'here', 'current', 'above', 'it', 'the file', 'the code']
            action_verbs = ['read', 'analyze', 'explain', 'review', 'check', 'look', 'help', 'understand', 'optimize', 'improve', 'debug', 'fix']
            
            # Check if the message seems to reference the active file
            seems_to_reference_file = (
                # Short questions like "What about this?" or "Can you read it?"
                (len(content_lower.split()) <= 8 and any(ref in content_lower for ref in file_reference_indicators)) or
                # Any action verb with a reference indicator
                (any(verb in content_lower for verb in action_verbs) and any(ref in content_lower for ref in file_reference_indicators)) or
                # Very short contextual questions
                content_lower in ['?', 'and this?', 'this one?', 'how about this?', 'what about this one?']
            )
            
            if seems_to_reference_file:
                file_info = f"\n\n**Current file: {active_file.get('file', 'unknown')}** ({active_file.get('language', 'unknown')} file)\n"
                file_info += f"```{active_file.get('language', '')}\n{active_file.get('content', '')}\n```"
                enhanced_content = f"{message_data.content}\n{file_info}"
    
    # Add workspace instructions if any
    enhanced_content = enhanced_content + workspace_instructions
    
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
                    limit=15,  # Increased from 5 to retrieve more relevant memories
                    use_bm25=True,
                    use_semantic=True
                )
            except (ImportError, AttributeError):
                # Fallback to basic service
                memory_service = ProjectMemoryService(project.id, project.name)
                memories = await memory_service.search_memories(
                    query=message_data.content,
                    conversation_id=None,  # Search across all conversations in the project
                    limit=20  # Further increased to retrieve more relevant memories for better context
                )
            
            if memories:
                logger.info(f"Found {len(memories)} memories for query: '{message_data.content[:50]}...'")
                memory_context = "\n\n=== Relevant Memories ===\n"
                for mem in memories[:10]:  # Use up to 10 memories in context (was 3)
                    memory_context += f"- {mem.content[:200]}...\n"
            else:
                logger.warning(f"No memories found for query: '{message_data.content[:50]}...'")
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
                message=enhanced_content,
                history=history,
                memory_context=memory_context,
                model=message_data.model,
                image_data=message_data.image_data if message_data.has_image else None,
                response_style=message_data.response_style,
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
            cost_threshold = 0.000001  # Ultra-low threshold to store virtually all responses
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