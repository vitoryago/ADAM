import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useTheme } from "@/components/theme-provider";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Brain, Plus, Moon, Sun, Settings, Trash2, X, MessageSquare, Edit2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Conversation, Project } from "@shared/schema";

interface SidebarProps {
  project: Project;
  isOpen: boolean;
  onClose: () => void;
  currentConversationId: string | null;
  onConversationSelect: (id: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ 
  project,
  isOpen, 
  onClose, 
  currentConversationId, 
  onConversationSelect, 
  onNewChat 
}: SidebarProps) {
  const { theme, toggleTheme } = useTheme();
  const { toast } = useToast();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const { data: conversations = [], isLoading } = useQuery<Conversation[]>({
    queryKey: ["/api/projects", project.id, "conversations"],
  });

  const deleteConversationMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/conversations/${id}`);
    },
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects", project.id, "conversations"] });
      // If we deleted the current conversation, navigate to a new chat
      if (currentConversationId === deletedId) {
        onNewChat();
      }
      toast({
        title: "Conversation deleted",
        description: "The conversation has been removed.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete conversation.",
        variant: "destructive",
      });
    },
  });

  const clearAllMutation = useMutation({
    mutationFn: async () => {
      await apiRequest("DELETE", `/api/projects/${project.id}/conversations`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects", project.id, "conversations"] });
      onNewChat();
      toast({
        title: "All conversations cleared",
        description: "Your chat history has been cleared.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to clear conversations.",
        variant: "destructive",
      });
    },
  });

  const renameConversationMutation = useMutation({
    mutationFn: async ({ id, title }: { id: string; title: string }) => {
      await apiRequest("PATCH", `/api/conversations/${id}`, { title });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/projects", project.id, "conversations"] });
      setEditingId(null);
      toast({
        title: "Conversation renamed",
        description: "The conversation title has been updated.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to rename conversation.",
        variant: "destructive",
      });
    },
  });

  const handleStartEdit = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setEditTitle(conversation.title);
  };

  const handleSaveEdit = (id: string) => {
    if (editTitle.trim()) {
      renameConversationMutation.mutate({ id, title: editTitle.trim() });
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle("");
  };

  const formatTimestamp = (date: Date | string | null) => {
    if (!date) return "";
    const d = new Date(date);
    const now = new Date();
    const diffHours = Math.floor((now.getTime() - d.getTime()) / (1000 * 60 * 60));
    
    if (diffHours < 1) return "Just now";
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffHours < 48) return "Yesterday";
    return d.toLocaleDateString();
  };

  return (
    <>
      <div className={cn(
        "w-80 bg-muted/30 border-r border-border flex flex-col transition-all duration-300 lg:translate-x-0 fixed lg:relative z-40 h-full",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-lg flex items-center justify-center">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <h1 className="text-xl font-bold">ADAM</h1>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={onClose}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          
          <Button 
            onClick={onNewChat}
            className="w-full bg-adam-primary hover:bg-adam-primary/90 text-white"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Chat
          </Button>
        </div>

        {/* Chat History */}
        <ScrollArea className="flex-1 p-4 custom-scrollbar">
          <div className="space-y-2">
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="p-3 rounded-lg bg-muted/50 animate-pulse">
                    <div className="h-4 bg-muted rounded w-3/4 mb-2" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                ))}
              </div>
            ) : conversations.length === 0 ? (
              <div className="text-center py-8">
                <MessageSquare className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground">No conversations yet</p>
                <p className="text-xs text-muted-foreground mt-1">Start a new chat to get started</p>
              </div>
            ) : (
              conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={cn(
                    "p-3 rounded-lg cursor-pointer transition-colors group hover:bg-muted relative",
                    currentConversationId === conversation.id && "bg-muted"
                  )}
                  onClick={() => editingId !== conversation.id && onConversationSelect(conversation.id)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      {editingId === conversation.id ? (
                        <input
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              handleSaveEdit(conversation.id);
                            } else if (e.key === "Escape") {
                              handleCancelEdit();
                            }
                          }}
                          onClick={(e) => e.stopPropagation()}
                          className="w-full px-2 py-1 text-sm bg-background border rounded"
                          autoFocus
                        />
                      ) : (
                        <>
                          <p className="text-sm font-medium truncate">
                            {conversation.title}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {formatTimestamp(conversation.updatedAt || conversation.createdAt)}
                          </p>
                        </>
                      )}
                    </div>
                    <div className={cn(
                      "flex items-center gap-1 flex-shrink-0 transition-opacity",
                      editingId === conversation.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                    )}>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (editingId === conversation.id) {
                            handleCancelEdit();
                          } else {
                            handleStartEdit(conversation);
                          }
                        }}
                        title="Edit conversation name"
                      >
                        {editingId === conversation.id ? (
                          <X className="w-4 h-4" />
                        ) : (
                          <Edit2 className="w-4 h-4" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 hover:bg-destructive/20 hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm('Are you sure you want to delete this conversation?')) {
                            deleteConversationMutation.mutate(conversation.id);
                          }
                        }}
                        title="Delete conversation"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              title="Toggle theme"
            >
              {theme === "light" ? (
                <Moon className="w-4 h-4" />
              ) : (
                <Sun className="w-4 h-4" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Settings"
            >
              <Settings className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Clear all conversations"
              onClick={() => clearAllMutation.mutate()}
              disabled={clearAllMutation.isPending || conversations.length === 0}
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
