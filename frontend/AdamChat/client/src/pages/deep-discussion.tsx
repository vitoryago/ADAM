import { useState } from "react";
import { useParams, useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { ConfigScreen } from "@/components/deep-discussion/config-screen";
import { LiveView } from "@/components/deep-discussion/live-view";
import { listSessions } from "@/lib/deep-discussion-api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface DeepDiscussionSession {
  id: string;
  question: string;
  pattern: string;
  status: string;
  created_at?: string;
}

export default function DeepDiscussion() {
  const { projectId, sessionId: routeSessionId } = useParams<{
    projectId: string;
    sessionId?: string;
  }>();
  const [, setLocation] = useLocation();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(
    routeSessionId || null,
  );
  const [replaySessionId, setReplaySessionId] = useState<string | null>(null);

  const { data: sessions = [], refetch: refetchSessions } = useQuery<
    DeepDiscussionSession[]
  >({
    queryKey: [`/api/deep-discussion/sessions`, projectId],
    queryFn: () => listSessions(projectId!),
    enabled: !!projectId,
  });

  const handleStart = (sessionId: string) => {
    setActiveSessionId(sessionId);
    setLocation(`/project/${projectId}/deep-discussion/${sessionId}`);
    refetchSessions();
  };

  const handleNewSession = () => {
    setActiveSessionId(null);
    setReplaySessionId(null);
    setLocation(`/project/${projectId}/deep-discussion`);
  };

  const handleReplay = () => {
    // Store the session being replayed so ConfigScreen can pre-fill
    setReplaySessionId(activeSessionId);
    setActiveSessionId(null);
    setLocation(`/project/${projectId}/deep-discussion`);
  };

  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId);
    setLocation(`/project/${projectId}/deep-discussion/${sessionId}`);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r border-border flex flex-col bg-background shrink-0">
        <div className="p-4 border-b border-border">
          <Button
            onClick={handleNewSession}
            className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
            size="sm"
          >
            + New Session
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="p-2">
            <p className="text-xs text-muted-foreground px-2 py-1 uppercase tracking-wide font-medium">
              Past Sessions
            </p>
            {sessions.length === 0 ? (
              <p className="text-xs text-muted-foreground px-2 py-2">
                No sessions yet.
              </p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => handleSelectSession(session.id)}
                  className={cn(
                    "w-full text-left rounded-md px-3 py-2 text-sm transition-colors",
                    "hover:bg-muted",
                    activeSessionId === session.id
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-foreground",
                  )}
                >
                  <div className="truncate">{session.question}</div>
                  <div className="text-xs text-muted-foreground capitalize mt-0.5">
                    {session.pattern} · {session.status}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="p-4 border-t border-border">
          <button
            onClick={() => setLocation(`/project/${projectId}`)}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            ← Back to Chat
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        {activeSessionId ? (
          <LiveView
            sessionId={activeSessionId}
            onReplay={handleReplay}
          />
        ) : (
          <ConfigScreen
            projectId={projectId!}
            onStart={handleStart}
          />
        )}
      </main>
    </div>
  );
}
