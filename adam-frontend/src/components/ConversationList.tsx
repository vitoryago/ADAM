import { useState } from 'react';
import type { Project, Conversation } from '../lib/api';

interface ConversationListProps {
  project: Project;
  conversations: Conversation[];
  selectedConversation: Conversation | null;
  onSelectConversation: (conversation: Conversation) => void;
  onCreateConversation: (title: string) => void;
  onDeleteConversation: (conversationId: string) => void;
  onTogglePin: (conversationId: string) => void;
}

export function ConversationList({
  project,
  conversations,
  selectedConversation,
  onSelectConversation,
  onCreateConversation,
  onDeleteConversation,
  onTogglePin,
}: ConversationListProps) {
  const [showNewModal, setShowNewModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');

  const handleCreateNew = () => {
    if (newTitle.trim()) {
      onCreateConversation(newTitle.trim());
      setNewTitle('');
      setShowNewModal(false);
    }
  };

  return (
    <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
      {/* Project Header */}
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white mb-2">{project.name}</h2>
        <p className="text-sm text-gray-400 mb-4">
          {project.description || 'No description'}
        </p>
        
        {/* New Conversation Button */}
        <button
          onClick={() => setShowNewModal(true)}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span>New Conversation</span>
        </button>
      </div>
      
      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
        <div className="space-y-2">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`p-3 rounded-lg hover:bg-gray-700 cursor-pointer transition-colors ${
                selectedConversation?.id === conv.id ? 'bg-gray-700' : ''
              }`}
              onClick={() => onSelectConversation(conv)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium text-white truncate">{conv.title}</h4>
                  <p className="text-xs text-gray-400 mt-1">
                    {conv.message_count} messages
                    {conv.is_pinned && <span className="ml-2">📌</span>}
                  </p>
                </div>
                <div className="flex items-center space-x-1 ml-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onTogglePin(conv.id);
                    }}
                    className="p-1 hover:bg-gray-600 rounded"
                    title={conv.is_pinned ? 'Unpin' : 'Pin'}
                  >
                    <svg
                      className={`w-4 h-4 ${
                        conv.is_pinned ? 'text-yellow-400' : 'text-gray-400'
                      }`}
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M5 5a2 2 0 012-2h6a2 2 0 012 2v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2V5z" />
                    </svg>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('Delete this conversation?')) {
                        onDeleteConversation(conv.id);
                      }
                    }}
                    className="p-1 hover:bg-gray-600 rounded text-red-400"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* New Conversation Modal */}
      {showNewModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-96">
            <h3 className="text-xl font-semibold text-white mb-4">New Conversation</h3>
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateNew()}
              placeholder="Enter conversation title..."
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              autoFocus
            />
            <div className="flex justify-end space-x-3 mt-4">
              <button
                onClick={() => {
                  setShowNewModal(false);
                  setNewTitle('');
                }}
                className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateNew}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}