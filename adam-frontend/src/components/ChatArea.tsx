import { useState, useEffect, useRef } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { messagesApi } from '../lib/api';
import type { Conversation, Message } from '../lib/api';

interface ChatAreaProps {
  conversation: Conversation | null;
  onConversationUpdate?: (conversation: Conversation) => void;
}

export function ChatArea({ conversation, onConversationUpdate }: ChatAreaProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [model, setModel] = useState<string>('');
  const [useMemory, setUseMemory] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (conversation) {
      loadMessages();
    } else {
      setMessages([]);
    }
  }, [conversation?.id]);

  const loadMessages = async () => {
    if (!conversation) return;
    
    try {
      setLoading(true);
      const response = await messagesApi.list(conversation.id);
      setMessages(response.data);
    } catch (err) {
      console.error('Error loading messages:', err);
    } finally {
      setLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (content: string, _imageData?: string) => {
    if (!conversation || streaming) return;

    // Add user message immediately
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      conversation_id: conversation.id,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    scrollToBottom();

    // Create assistant message placeholder
    const assistantMessage: Message = {
      id: `temp-assistant-${Date.now()}`,
      conversation_id: conversation.id,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, assistantMessage]);
    setStreaming(true);

    try {
      // Use fetch for streaming
      const response = await messagesApi.stream(conversation.id, content, {
        model: model || undefined,
        use_memory: useMemory,
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';

      while (reader) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'assistant_chunk') {
                fullContent += data.content;
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessage.id
                      ? { ...msg, content: fullContent }
                      : msg
                  )
                );
                scrollToBottom();
              } else if (data.type === 'complete') {
                // Update with final metadata
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === assistantMessage.id
                      ? {
                          ...msg,
                          content: fullContent,
                          model: data.model,
                          tokens_used: data.tokens,
                          cost: data.cost,
                        }
                      : msg
                  )
                );
                
                // Update conversation message count
                if (onConversationUpdate) {
                  onConversationUpdate({
                    ...conversation,
                    message_count: conversation.message_count + 2,
                  });
                }
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Streaming error');
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (err) {
      console.error('Error sending message:', err);
      // Remove the assistant message on error
      setMessages((prev) => prev.filter((msg) => msg.id !== assistantMessage.id));
    } finally {
      setStreaming(false);
      scrollToBottom();
    }
  };

  if (!conversation) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <svg
            className="w-16 h-16 mx-auto mb-4"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z"
              clipRule="evenodd"
            />
          </svg>
          <p>Select or create a conversation to start chatting</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Chat Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-semibold text-white">{conversation.title}</h3>
          
          {/* Chat Controls */}
          <div className="flex items-center space-x-4">
            {/* Model Selector */}
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">🤖 Automatic</option>
              <option value="grok-3-mini-high">⚡ Grok-3 Mini</option>
              <option value="grok-4">🧠 Grok-4</option>
              <option value="grok-4-reasoning">🎯 Grok-4 Reasoning</option>
            </select>
            
            {/* Memory Toggle */}
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={useMemory}
                onChange={(e) => setUseMemory(e.target.checked)}
                className="sr-only"
              />
              <div className="relative">
                <div className="block bg-gray-600 w-10 h-6 rounded-full"></div>
                <div
                  className={`absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${
                    useMemory ? 'transform translate-x-4 bg-blue-500' : ''
                  }`}
                ></div>
              </div>
              <span className="ml-2 text-sm text-white">Memory</span>
            </label>
          </div>
        </div>
      </div>
      
      {/* Messages */}
      <MessageList messages={messages} loading={loading} />
      <div ref={messagesEndRef} />
      
      {/* Input */}
      <MessageInput
        onSendMessage={handleSendMessage}
        disabled={streaming || !conversation}
      />
    </div>
  );
}