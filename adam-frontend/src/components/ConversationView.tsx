import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ConversationList } from './ConversationList';
import { ChatArea } from './ChatArea';
import { projectsApi, conversationsApi } from '../lib/api';
import type { Project, Conversation } from '../lib/api';

export function ConversationView() {
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (projectId) {
      loadProjectAndConversations();
    }
  }, [projectId]);

  const loadProjectAndConversations = async () => {
    if (!projectId) return;
    
    try {
      setLoading(true);
      // Load project details
      const projectResponse = await projectsApi.get(projectId);
      setProject(projectResponse.data);
      
      // Load conversations
      const conversationsResponse = await conversationsApi.list(projectId);
      setConversations(conversationsResponse.data);
      
      // Select first conversation if available
      if (conversationsResponse.data.length > 0) {
        setSelectedConversation(conversationsResponse.data[0]);
      }
    } catch (err) {
      setError('Failed to load project data');
      console.error('Error loading project:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateConversation = async (title: string) => {
    if (!projectId) return;
    
    try {
      const response = await conversationsApi.create(projectId, title);
      const newConversation = response.data;
      setConversations([newConversation, ...conversations]);
      setSelectedConversation(newConversation);
    } catch (err) {
      console.error('Error creating conversation:', err);
    }
  };

  const handleDeleteConversation = async (conversationId: string) => {
    try {
      await conversationsApi.delete(conversationId);
      setConversations(conversations.filter(c => c.id !== conversationId));
      if (selectedConversation?.id === conversationId) {
        setSelectedConversation(conversations.find(c => c.id !== conversationId) || null);
      }
    } catch (err) {
      console.error('Error deleting conversation:', err);
    }
  };

  const handleTogglePin = async (conversationId: string) => {
    try {
      await conversationsApi.togglePin(conversationId);
      // Reload conversations to get updated pin status
      if (projectId) {
        const response = await conversationsApi.list(projectId);
        setConversations(response.data);
      }
    } catch (err) {
      console.error('Error toggling pin:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-400">Loading project...</div>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-red-400">{error || 'Project not found'}</div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Conversation Sidebar */}
      <ConversationList
        project={project}
        conversations={conversations}
        selectedConversation={selectedConversation}
        onSelectConversation={setSelectedConversation}
        onCreateConversation={handleCreateConversation}
        onDeleteConversation={handleDeleteConversation}
        onTogglePin={handleTogglePin}
      />
      
      {/* Chat Area */}
      <ChatArea
        conversation={selectedConversation}
        onConversationUpdate={(updatedConversation) => {
          setConversations(conversations.map(c => 
            c.id === updatedConversation.id ? updatedConversation : c
          ));
        }}
      />
    </div>
  );
}