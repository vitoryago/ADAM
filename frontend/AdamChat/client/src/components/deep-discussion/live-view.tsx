import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, startSession } from "@/lib/deep-discussion-api";
import { ProgressBar } from "./progress-bar";
import { AgentCard, type AgentCardProps } from "./agent-card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface LiveViewProps {
  sessionId: string;
  onReplay: () => void;
}

interface AgentState extends AgentCardProps {
  id: string;
}

interface SessionInfo {
  question: string;
  pattern: string;
  status: string;
}

const PATTERN_DISPLAY_NAMES: Record<string, string> = {
  sequential: "Sequential",
  debate: "Debate",
  peer_review: "Peer Review",
};

export function LiveView({ sessionId, onReplay }: LiveViewProps) {
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [currentStep, setCurrentStep] = useState<string>("");
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  const [sessionStatus, setSessionStatus] = useState<
    "loading" | "running" | "complete" | "error"
  >("loading");
  const [error, setError] = useState<string | null>(null);
  const [totalCost, setTotalCost] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);

  // Derive total cost from all agent costs
  const derivedTotalCost = agents.reduce((sum, a) => sum + (a.cost ?? 0), 0);

  const handleSSEMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      const { type } = data;

      if (type === "session_start") {
        setSessionInfo((prev) =>
          prev
            ? { ...prev, pattern: data.pattern ?? prev.pattern }
            : { question: data.question ?? "", pattern: data.pattern ?? "", status: "running" },
        );
        setSessionStatus("running");
      } else if (type === "step_start") {
        setCurrentStep(data.step ?? "");
      } else if (type === "agent_start") {
        const newAgent: AgentState = {
          id: data.agent_id ?? `agent-${Date.now()}`,
          role: data.role ?? "reasoner",
          name: data.name ?? data.role ?? "Agent",
          model: data.model ?? "",
          status: "thinking",
          content: "",
          cost: 0,
          tokens: 0,
          badge: data.badge ?? "",
        };
        setAgents((prev) => [...prev, newAgent]);
      } else if (type === "agent_chunk") {
        const agentId = data.agent_id;
        const chunk: string = data.chunk ?? "";
        setAgents((prev) =>
          prev.map((a) =>
            a.id === agentId ? { ...a, content: a.content + chunk } : a,
          ),
        );
      } else if (type === "agent_done") {
        const agentId = data.agent_id;
        setAgents((prev) =>
          prev.map((a) =>
            a.id === agentId
              ? {
                  ...a,
                  status: "done",
                  cost: data.cost ?? a.cost,
                  tokens: data.tokens ?? a.tokens,
                }
              : a,
          ),
        );
      } else if (type === "step_complete") {
        const step: string = data.step ?? "";
        setCompletedSteps((prev) => (prev.includes(step) ? prev : [...prev, step]));
        setCurrentStep((prev) => (prev === step ? "" : prev));
      } else if (type === "session_complete") {
        setFinalAnswer(data.answer ?? data.result ?? null);
        setSessionStatus("complete");
        setCurrentStep("");
        // Close SSE
        eventSourceRef.current?.close();
      } else if (type === "agent_error") {
        const agentId = data.agent_id;
        setAgents((prev) =>
          prev.map((a) =>
            a.id === agentId ? { ...a, status: "error", content: a.content + (data.error ? `\n\nError: ${data.error}` : "") } : a,
          ),
        );
      } else if (type === "session_error") {
        setError(data.error ?? "An unknown error occurred.");
        setSessionStatus("error");
        eventSourceRef.current?.close();
      }
    } catch {
      // non-JSON or empty keepalive — ignore
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const session = await getSession(sessionId);
        if (cancelled) return;

        setSessionInfo({
          question: session.question ?? "",
          pattern: session.pattern ?? "sequential",
          status: session.status ?? "pending",
        });

        // If session already completed, just show the stored data
        if (session.status === "complete" || session.status === "completed") {
          setFinalAnswer(session.result ?? session.answer ?? null);
          setSessionStatus("complete");
          // Reconstruct agents from session data if available
          if (Array.isArray(session.entries)) {
            const reconstructed: AgentState[] = session.entries.map(
              (e: Record<string, unknown>, idx: number) => ({
                id: `entry-${idx}`,
                role: (e.role as string) ?? "reasoner",
                name: (e.name as string) ?? (e.role as string) ?? "Agent",
                model: (e.model as string) ?? "",
                status: "done" as const,
                content: (e.content as string) ?? "",
                cost: (e.cost as number) ?? 0,
                tokens: (e.tokens as number) ?? 0,
                badge: (e.badge as string) ?? "",
              }),
            );
            setAgents(reconstructed);
          }
          return;
        }

        // Start SSE stream
        const es = startSession(sessionId);
        eventSourceRef.current = es;

        es.onmessage = handleSSEMessage;
        es.onerror = () => {
          if (!cancelled) {
            setError("Connection lost. The session may have ended.");
            setSessionStatus("error");
          }
        };
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load session.");
          setSessionStatus("error");
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
    };
  }, [sessionId, handleSSEMessage]);

  if (sessionStatus === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-muted-foreground">
          <span className="w-5 h-5 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          <p className="text-sm">Loading session…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 max-w-3xl mx-auto py-6 px-4">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <h2 className="text-lg font-bold leading-snug">
              {sessionInfo?.question ?? "Deep Discussion"}
            </h2>
            <div className="flex items-center gap-3 mt-1 flex-wrap">
              <span className="text-xs rounded-full bg-muted px-2 py-0.5 text-muted-foreground capitalize">
                {PATTERN_DISPLAY_NAMES[sessionInfo?.pattern ?? ""] ?? sessionInfo?.pattern}
              </span>
              <span
                className={cn(
                  "text-xs font-medium",
                  sessionStatus === "running" && "text-amber-600",
                  sessionStatus === "complete" && "text-emerald-600",
                  sessionStatus === "error" && "text-destructive",
                )}
              >
                {sessionStatus === "running" && "Running…"}
                {sessionStatus === "complete" && "Complete"}
                {sessionStatus === "error" && "Error"}
              </span>
              {derivedTotalCost > 0 && (
                <span className="text-xs font-mono text-muted-foreground">
                  ${derivedTotalCost.toFixed(4)} total
                </span>
              )}
            </div>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={onReplay}
            className="shrink-0"
          >
            Replay with Changes
          </Button>
        </div>
      </div>

      {/* Progress bar */}
      {sessionInfo?.pattern && (
        <ProgressBar
          pattern={sessionInfo.pattern}
          currentStep={currentStep}
          completedSteps={completedSteps}
        />
      )}

      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3">
          <p className="text-sm text-destructive font-medium">Error</p>
          <p className="text-sm text-destructive/80 mt-0.5">{error}</p>
        </div>
      )}

      {/* Final answer */}
      {finalAnswer && (
        <div className="rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-4">
          <p className="text-xs font-semibold text-emerald-700 uppercase tracking-wide mb-2">
            Final Answer
          </p>
          <pre className="text-sm whitespace-pre-wrap break-words font-sans leading-relaxed text-foreground">
            {finalAnswer}
          </pre>
        </div>
      )}

      {/* Agent cards */}
      {agents.length > 0 && (
        <div className="flex flex-col gap-3">
          {agents.map((agent) => (
            <AgentCard
              key={agent.id}
              role={agent.role}
              name={agent.name}
              model={agent.model}
              status={agent.status}
              content={agent.content}
              cost={agent.cost}
              tokens={agent.tokens}
              badge={agent.badge}
            />
          ))}
        </div>
      )}

      {/* Empty state when running but no agents yet */}
      {sessionStatus === "running" && agents.length === 0 && !error && (
        <div className="flex items-center gap-2 text-muted-foreground text-sm">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          <span>Waiting for agents to start…</span>
        </div>
      )}
    </div>
  );
}
