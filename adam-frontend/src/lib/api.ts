import axios from 'axios';

// Types
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

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth (future use)
api.interceptors.request.use(
  (config) => {
    // Add auth token here when we implement authentication
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
    }
    return Promise.reject(error);
  }
);

// API Methods
export const projectsApi = {
  list: () => api.get<Project[]>('/projects'),
  get: (id: string) => api.get<Project>(`/projects/${id}`),
  create: (data: Partial<Project>) => api.post<Project>('/projects', data),
  update: (id: string, data: Partial<Project>) => api.patch<Project>(`/projects/${id}`, data),
  delete: (id: string) => api.delete(`/projects/${id}`),
};

export const conversationsApi = {
  list: (projectId: string) => api.get<Conversation[]>(`/projects/${projectId}/conversations`),
  get: (id: string) => api.get<Conversation>(`/conversations/${id}`),
  create: (projectId: string, title: string) => 
    api.post<Conversation>(`/projects/${projectId}/conversations`, { title }),
  delete: (id: string) => api.delete(`/conversations/${id}`),
  togglePin: (id: string) => api.post(`/conversations/${id}/toggle-pin`),
};

export const messagesApi = {
  list: (conversationId: string) => 
    api.get<Message[]>(`/conversations/${conversationId}/messages`),
  send: (conversationId: string, content: string, options?: {
    model?: string;
    use_memory?: boolean;
  }) => 
    api.post<Message[]>(`/conversations/${conversationId}/messages`, {
      content,
      ...options,
    }),
  stream: (conversationId: string, content: string, options?: {
    model?: string;
    use_memory?: boolean;
  }) => {
    // For streaming, we'll use fetch API instead of axios
    return fetch(`http://localhost:8000/api/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        content,
        ...options,
      }),
    });
  },
};

export default api;