import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./message-bubble";
import { MessageInput } from "./message-input";
import { TypingIndicator } from "./typing-indicator";
import { ModelSelector } from "./model-selector";
import { SearchToggle } from "./search-toggle";
import { useWebSocket } from "@/hooks/use-websocket";
import { useTheme } from "@/components/theme-provider";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Brain, Menu, Moon, Sun } from "lucide-react";
import type { Message, Conversation, Project } from "@shared/schema";
import { transformMessage } from "@/lib/api-types";

interface ChatAreaProps {
  project: Project;
  conversationId: string | null;
  onConversationCreated: (id: string) => void;
  onToggleSidebar: () => void;
}

export function ChatArea({ project, conversationId, onConversationCreated, onToggleSidebar }: ChatAreaProps) {
  const { theme, toggleTheme } = useTheme();
  const { toast } = useToast();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("automatic");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [searchMode, setSearchMode] = useState<string>("auto");
  
  const { isConnected, isTyping, error, sendMessage, onMessage, reconnect } = useWebSocket();

  // Load messages for current conversation
  const { data: conversationMessages = [], isLoading } = useQuery<Message[]>({
    queryKey: ["/api/conversations", conversationId, "messages"],
    enabled: !!conversationId,
    select: (data: any[]) => data.map((msg) => {
      const transformed = transformMessage(msg);
      // If the message has an image_url from backend, use it as preview
      if (msg.has_image && msg.image_url) {
        transformed.metadata = {
          ...transformed.metadata,
          has_image: true,
          preview: msg.image_url.startsWith('data:') ? msg.image_url : `data:image/jpeg;base64,${msg.image_url}`
        };
      }
      return transformed;
    })
  });

  // Update local messages when conversation changes
  useEffect(() => {
    setMessages(conversationMessages);
  }, [conversationMessages]);

  // Listen for new WebSocket messages
  useEffect(() => {
    const unsubscribe = onMessage((newMessage) => {
      setMessages(prev => [...prev, newMessage]);
    });
    return unsubscribe;
  }, [onMessage]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Show connection error toast
  useEffect(() => {
    if (error) {
      toast({
        title: "Connection Error",
        description: error,
        variant: "destructive",
        action: (
          <Button variant="outline" size="sm" onClick={reconnect}>
            Retry
          </Button>
        ),
      });
    }
  }, [error, toast, reconnect]);

  const createConversationMutation = useMutation({
    mutationFn: async (title: string) => {
      const response = await apiRequest("POST", `/api/projects/${project.id}/conversations`, {
        title
      });
      return await response.json() as Conversation;
    },
    onSuccess: (conversation) => {
      onConversationCreated(conversation.id);
      queryClient.invalidateQueries({ queryKey: ["/api/projects", project.id, "conversations"] });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to create new conversation.",
        variant: "destructive",
      });
    },
  });

  const handleSendMessage = async (content: string, attachedFile?: any) => {
    if (!isConnected) {
      toast({
        title: "Not Connected",
        description: "Please wait for connection to be established.",
        variant: "destructive",
      });
      return;
    }

    let targetConversationId = conversationId;

    // Create new conversation if needed
    if (!targetConversationId) {
      try {
        const title = content.length > 50 ? content.substring(0, 47) + "..." : content;
        const conversation = await createConversationMutation.mutateAsync(title);
        targetConversationId = conversation.id;
      } catch (error) {
        console.error("Failed to create conversation:", error);
        return;
      }
    }

    // Add user message immediately for better UX
    const tempUserMessage: Message = {
      id: `temp-${Date.now()}`,
      conversationId: targetConversationId,
      content: content || (attachedFile ? `[Attached: ${attachedFile.name}]` : ''),
      role: 'user',
      createdAt: new Date(),
      metadata: attachedFile ? {
        has_image: attachedFile.type.startsWith('image/'),
        file_name: attachedFile.name,
        file_type: attachedFile.type,
        file_size: attachedFile.size,
        preview: attachedFile.preview
      } : undefined
    };
    
    setMessages(prev => [...prev, tempUserMessage]);
    
    // Auto-select vision model for images if automatic mode
    let modelToUse = selectedModel;
    if (attachedFile?.type.startsWith('image/') && selectedModel === "automatic") {
      modelToUse = "grok-2-vision-1212"; // Use vision model for images
    }

    // Send message via REST API
    setIsProcessing(true);
    try {
      const requestBody: any = {
        content,
        use_memory: true,
        model: modelToUse === "automatic" ? null : modelToUse,
        use_search: searchEnabled,
        search_mode: searchEnabled ? searchMode : null
      };

      // Handle file attachments
      if (attachedFile) {
        if (attachedFile.type.startsWith('image/')) {
          // For images, send as image data
          requestBody.has_image = true;
          requestBody.image_data = attachedFile.data;
        } else {
          // For non-images, decode base64 and include content in message
          try {
            const decodedContent = atob(attachedFile.data);
            const fileExtension = attachedFile.name.split('.').pop() || 'txt';
            const prefix = content ? `${content}\n\n` : '';
            requestBody.content = `${prefix}File: ${attachedFile.name}\n\`\`\`${fileExtension}\n${decodedContent}\n\`\`\``;
          } catch (e) {
            // If decoding fails, just mention the file
            const prefix = content ? `${content}\n\n` : '';
            requestBody.content = `${prefix}[Attached file: ${attachedFile.name}]`;
          }
        }
      }

      // Use streaming endpoint for better UX
      const response = await fetch(`/api/conversations/${targetConversationId}/messages/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      // Add assistant message placeholder
      const assistantMessageId = `assistant-${Date.now()}`;
      const assistantMessage: Message = {
        id: assistantMessageId,
        conversationId: targetConversationId,
        content: '',
        role: 'assistant',
        createdAt: new Date(),
        metadata: {}
      };
      
      setMessages(prev => {
        // Remove temp user message and add real user + streaming assistant
        const filtered = prev.filter(msg => msg.id !== tempUserMessage.id);
        const userMsg = { ...tempUserMessage, id: `user-${Date.now()}` };
        if (attachedFile?.preview) {
          userMsg.metadata = {
            ...userMsg.metadata,
            file_name: attachedFile.name,
            file_type: attachedFile.type,
            file_size: attachedFile.size,
            preview: attachedFile.preview
          };
        }
        return [...filtered, userMsg, assistantMessage];
      });
      
      // Remove processing indicator since streaming has started
      setIsProcessing(false);

      if (reader) {
        let accumulatedContent = '';
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          
          // Process complete lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer
          
          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data: ')) {
              try {
                const jsonStr = trimmed.slice(6);
                if (jsonStr.trim()) {
                  const data = JSON.parse(jsonStr);
                  
                  if (data.type === 'assistant_chunk' && data.content) {
                    accumulatedContent += data.content;
                    // Update assistant message with accumulated content
                    setMessages(prev => prev.map(msg => 
                      msg.id === assistantMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  } else if (data.type === 'complete') {
                    // Update final metadata
                    setMessages(prev => prev.map(msg => 
                      msg.id === assistantMessageId 
                        ? { 
                            ...msg, 
                            metadata: { 
                              ...msg.metadata, 
                              model: data.model,
                              tokens_used: data.tokens,
                              cost: data.cost
                            } 
                          }
                        : msg
                    ));
                  } else if (data.type === 'error') {
                    console.error('Streaming error:', data.message);
                    toast({
                      title: "Streaming Error",
                      description: data.message,
                      variant: "destructive",
                    });
                  }
                }
              } catch (e) {
                console.error('Failed to parse streaming data:', e, 'Line:', trimmed);
              }
            }
          }
        }
        
        // Process any remaining buffer
        if (buffer.trim()) {
          console.warn('Incomplete streaming data in buffer:', buffer);
        }
      }
      
      // Refresh the conversation messages
      queryClient.invalidateQueries({ 
        queryKey: ["/api/conversations", targetConversationId, "messages"] 
      });
    } catch (error) {
      console.error("Failed to send message:", error);
      // Remove the temporary message on error
      setMessages(prev => prev.filter(msg => msg.id !== tempUserMessage.id));
      
      // Better error message
      let errorMessage = "Failed to send message. Please try again.";
      if (error instanceof Error) {
        errorMessage = error.message;
        // Log more details for debugging
        console.error("Error details:", {
          message: error.message,
          stack: error.stack,
          attachedFile: attachedFile ? {
            name: attachedFile.name,
            type: attachedFile.type,
            size: attachedFile.size,
            dataLength: attachedFile.data?.length
          } : null
        });
      }
      
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Mobile Header */}
      <div className="lg:hidden flex items-center justify-between p-4 border-b border-border bg-background">
        <Button variant="ghost" size="icon" onClick={onToggleSidebar}>
          <Menu className="w-4 h-4" />
        </Button>
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-md flex items-center justify-center">
            <Brain className="w-3 h-3 text-white" />
          </div>
          <h1 className="text-lg font-bold">ADAM</h1>
        </div>
        <Button variant="ghost" size="icon" onClick={toggleTheme}>
          {theme === "light" ? (
            <Moon className="w-4 h-4" />
          ) : (
            <Sun className="w-4 h-4" />
          )}
        </Button>
      </div>

      {/* Messages Container */}
      <ScrollArea className="flex-1">
        <div className="max-w-4xl mx-auto px-4 py-8">
          {!conversationId && messages.length === 0 ? (
            // Welcome Screen
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-2xl mx-auto mb-4 flex items-center justify-center">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-2xl font-bold mb-2">Welcome to ADAM</h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                Your AI assistant ready to help with questions, analysis, and creative tasks. 
                How can I assist you today?
              </p>
            </div>
          ) : (
            // Messages
            <div className="space-y-6">
              {isLoading ? (
                <div className="space-y-6">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="flex justify-start animate-pulse">
                      <div className="flex items-start space-x-3 max-w-2xl">
                        <div className="w-8 h-8 bg-muted rounded-full" />
                        <div className="bg-muted rounded-2xl rounded-tl-md px-4 py-3 w-64 h-20" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))
              )}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="flex items-start space-x-3 max-w-2xl">
                    <div className="w-8 h-8 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-full flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-muted rounded-2xl rounded-tl-md px-4 py-3">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-2 h-2 bg-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-2 h-2 bg-foreground/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-sm text-muted-foreground">ADAM is thinking...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {isTyping && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Connection Status */}
      {!isConnected && (
        <div className="px-4 py-2 bg-destructive text-destructive-foreground text-sm text-center">
          Disconnected from server. Attempting to reconnect...
        </div>
      )}

      {/* Model Selector and Search Toggle */}
      <div className="px-4 py-2 border-t border-border">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-4">
            <div className="text-sm text-muted-foreground">Model:</div>
            <ModelSelector 
              value={selectedModel} 
              onChange={setSelectedModel} 
              disabled={!isConnected || isProcessing}
            />
          </div>
          <SearchToggle
            enabled={searchEnabled}
            onEnabledChange={setSearchEnabled}
            searchMode={searchMode}
            onSearchModeChange={setSearchMode}
            disabled={!isConnected || isProcessing}
          />
        </div>
      </div>

      {/* Message Input */}
      <MessageInput 
        onSendMessage={handleSendMessage} 
        disabled={!isConnected}
        conversationId={conversationId || undefined}
        model={selectedModel}
        useSearch={searchEnabled}
        onConversationComplete={() => {
          // Refresh messages when voice conversation completes
          queryClient.invalidateQueries({ 
            queryKey: ["/api/conversations", conversationId, "messages"] 
          });
        }}
      />
    </div>
  );
}
