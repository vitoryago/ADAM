import { useQuery } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  model?: string;
  tokens_used?: number;
  cost?: number;
}

interface ConversationData {
  id: string;
  name: string;
  project_id: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export function useConversation(conversationId: string | null) {
  const query = useQuery({
    queryKey: ["conversation", conversationId, "messages"],
    queryFn: async () => {
      if (!conversationId) return null;
      
      const response = await fetch(`/api/conversations/${conversationId}/messages`, {
        credentials: "include",
      });
      
      if (!response.ok) {
        throw new Error("Failed to fetch conversation");
      }
      
      return response.json() as Promise<Message[]>;
    },
    enabled: !!conversationId,
    refetchInterval: false,
  });

  return {
    messages: query.data || [],
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}

export function invalidateConversation(conversationId: string) {
  queryClient.invalidateQueries({
    queryKey: ["conversation", conversationId, "messages"],
  });
}