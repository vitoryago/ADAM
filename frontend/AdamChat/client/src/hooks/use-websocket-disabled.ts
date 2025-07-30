import { useState, useCallback } from "react";
import type { Message } from "@shared/schema";

// Disabled WebSocket hook - ADAM backend doesn't have WebSocket support yet
export function useWebSocket() {
  const [isConnected] = useState(false);
  const [isTyping] = useState(false);
  const [error] = useState<string | null>(null);

  const sendMessage = useCallback(async (message: Partial<Message>) => {
    console.log('WebSocket disabled - messages sent via REST API');
  }, []);

  const onMessage = useCallback((handler: (message: Message) => void) => {
    // No-op for now
    return () => {};
  }, []);

  const reconnect = useCallback(() => {
    console.log('WebSocket disabled - using REST API only');
  }, []);

  return {
    isConnected: true, // Pretend we're connected
    isTyping,
    error,
    sendMessage,
    onMessage,
    reconnect,
  };
}