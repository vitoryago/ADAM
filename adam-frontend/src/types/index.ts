// Shared types for ADAM v2

export interface Project {
  id: string;
  name: string;
  description?: string;
  settings: Record<string, any>;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  project_id: string;
  title: string;
  is_pinned: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  tokens_used?: number;
  cost?: number;
  created_at: string;
}