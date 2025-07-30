import { useState, useEffect } from "react";
import { useParams } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "@/components/chat/sidebar";
import { ChatArea } from "@/components/chat/chat-area";
import { ProjectHeader } from "@/components/project/project-header";
import { useIsMobile } from "@/hooks/use-mobile";
import { useLocation } from "wouter";
import type { Project } from "@shared/schema";

export default function Chat() {
  const { projectId, conversationId } = useParams();
  const [, setLocation] = useLocation();
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(conversationId || null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const isMobile = useIsMobile();

  const { data: project, isLoading: projectLoading } = useQuery<Project>({
    queryKey: ["/api/projects", projectId],
    enabled: !!projectId,
  });

  useEffect(() => {
    if (conversationId) {
      setCurrentConversationId(conversationId);
    }
  }, [conversationId]);

  if (projectLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-adam-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading project...</p>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Project not found</h2>
          <p className="text-muted-foreground mb-4">The project you're looking for doesn't exist.</p>
          <button
            onClick={() => setLocation("/")}
            className="px-4 py-2 bg-adam-primary text-white rounded-lg hover:bg-adam-primary/90"
          >
            Back to Projects
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background flex-col">
      {/* Project Header */}
      <ProjectHeader 
        project={project} 
        onBack={() => setLocation("/")} 
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          project={project}
          isOpen={isMobile ? isSidebarOpen : true}
          onClose={() => setIsSidebarOpen(false)}
          currentConversationId={currentConversationId}
          onConversationSelect={(id) => {
            setCurrentConversationId(id);
            setLocation(`/project/${projectId}/chat/${id}`);
            if (isMobile) {
              setIsSidebarOpen(false);
            }
          }}
          onNewChat={() => {
            setCurrentConversationId(null);
            setLocation(`/project/${projectId}`);
            if (isMobile) {
              setIsSidebarOpen(false);
            }
          }}
        />

        {/* Main Chat Area */}
        <ChatArea
          project={project}
          conversationId={currentConversationId}
          onConversationCreated={(id) => {
            setCurrentConversationId(id);
            setLocation(`/project/${projectId}/chat/${id}`);
          }}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        {/* Mobile overlay */}
        {isMobile && isSidebarOpen && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 z-30"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
