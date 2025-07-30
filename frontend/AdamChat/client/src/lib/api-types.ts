// API types for message responses with metadata

export interface MessageMetadata {
  model?: string;
  tokens_used?: number;
  cost?: number;
  has_image?: boolean;
  file_name?: string;
  file_type?: string;
  file_size?: number;
  preview?: string;
}

export interface MessageWithMetadata {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: MessageMetadata;
}

// Transform backend message format to frontend format
export function transformMessage(backendMessage: any): MessageWithMetadata {
  return {
    id: backendMessage.id,
    conversationId: backendMessage.conversation_id || backendMessage.conversationId,
    role: backendMessage.role,
    content: backendMessage.content,
    timestamp: backendMessage.created_at || backendMessage.timestamp,
    metadata: {
      model: backendMessage.model,
      tokens_used: backendMessage.tokens_used,
      cost: backendMessage.cost,
      has_image: backendMessage.has_image
    }
  };
}