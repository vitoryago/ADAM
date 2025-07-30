import { useState, useCallback } from "react";
import type { Message } from "@shared/schema";

// Disabled WebSocket hook - ADAM backend uses REST API only
export function useWebSocket() {
  const [isConnected] = useState(true); // Always "connected" for REST API
  const [isTyping] = useState(false);
  const [error] = useState<string | null>(null);

  const sendMessage = useCallback((conversationId: string, content: string) => {
    console.log('WebSocket disabled - messages are sent via REST API');
    // The actual message sending is handled by the chat component using REST API
  }, []);

  const onMessage = useCallback((handler: (message: Message) => void) => {
    // No-op - messages come from REST API queries
    return () => {};
  }, []);

  const reconnect = useCallback(() => {
    console.log('WebSocket disabled - using REST API');
  }, []);

  return {
    isConnected,
    isTyping,
    error,
    sendMessage,
    onMessage,
    reconnect
  };
}