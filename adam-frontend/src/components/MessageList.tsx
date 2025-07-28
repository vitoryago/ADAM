import type { Message } from '../lib/api';
import { MessageItem } from './MessageItem';

interface MessageListProps {
  messages: Message[];
  loading?: boolean;
}

export function MessageList({ messages, loading }: MessageListProps) {
  if (loading) {
    return (
      <div className="flex-1 overflow-y-auto p-6">
        <div className="animate-pulse space-y-4">
          <div className="flex justify-end">
            <div className="bg-gray-700 rounded-lg p-4 max-w-2xl w-full">
              <div className="h-4 bg-gray-600 rounded w-3/4"></div>
            </div>
          </div>
          <div className="flex">
            <div className="bg-gray-800 rounded-lg p-4 max-w-2xl w-full">
              <div className="h-4 bg-gray-700 rounded w-full mb-2"></div>
              <div className="h-4 bg-gray-700 rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </div>
  );
}