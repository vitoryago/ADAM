import { useEffect, useRef, useState, useCallback } from "react";
import { getSession, startSession, stopSession } from "@/lib/deep-discussion-api";
import { ProgressBar } from "./progress-bar";
import { AgentCard, type AgentCardProps } from "./agent-card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MessageContent } from "@/lib/message-parser";

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

function getPatternCompletedSteps(pattern: string): string[] {
  if (pattern === "debate") return ["debate_a", "debate_b", "rebuttal", "reconcile"];
  if (pattern === "peer_review") return ["produce", "review", "rebuttal", "react", "synthesize"];
  return ["produce", "code", "review", "synthesize"];
}

const PATTERN_DISPLAY_NAMES: Record<string, string> = {
  sequential: "Sequential",
  debate: "Debate",
  peer_review: "Peer Review",
};

const PATTERN_DESCRIPTIONS: Record<string, string> = {
  sequential:
    "Reasoner frames the approach, Coder expands it, Critic challenges it, and Synthesizer consolidates it.",
  debate:
    "Perspective A opens, Perspective B counters, Perspective A rebuts, and the Reconciler integrates the strongest points.",
  peer_review:
    "A producer draft is reviewed, rebutted, reacted to, and then synthesized into one answer.",
};

const STEP_NAME_TO_ID: Record<string, string> = {
  PRODUCE: "produce",
  REVIEW: "review",
  REBUTTAL: "rebuttal",
  REACT: "react",
  SYNTHESIZE: "synthesize",
};

const SEQUENTIAL_AGENT_STEP_IDS: Record<string, string> = {
  Reasoner: "produce",
  Coder: "code",
  Critic: "review",
  Synthesizer: "synthesize",
};

const DEBATE_AGENT_STEP_IDS: Record<string, string> = {
  "Perspective A": "debate_a",
  "Perspective B": "debate_b",
  "Perspective A Rebuttal": "rebuttal",
  Reconciler: "reconcile",
};

function normalizeStepId(step: unknown, name?: unknown): string {
  if (typeof name === "string" && STEP_NAME_TO_ID[name]) {
    return STEP_NAME_TO_ID[name];
  }
  if (typeof step === "string") {
    return STEP_NAME_TO_ID[step.toUpperCase()] ?? step.toLowerCase();
  }
  if (typeof step === "number") {
    return String(step);
  }
  return "";
}

function getAgentBadge(role: unknown, entryType?: unknown): string {
  if ((entryType ?? "").toString().toLowerCase() === "rebuttal") {
    return "REBUTTAL";
  }
  switch ((role ?? "").toString().toLowerCase()) {
    case "reasoner":
    case "coder":
      return "PRODUCER";
    case "critic":
      return "REVIEWER";
    case "synthesizer":
      return "FINAL";
    default:
      return "";
  }
}

function buildAgentsFromScratchpad(session: Record<string, unknown>): AgentState[] {
  const scratchpad = session.scratchpad_data as
    | { entries?: Array<Record<string, unknown>> }
    | undefined;
  const entries = scratchpad?.entries;

  if (!Array.isArray(entries)) {
    return [];
  }

  return entries.map((entry, idx) => ({
    id: (entry.agent_name as string) ?? `entry-${idx}`,
    role: ((entry.agent_role as string) ?? "reasoner").toLowerCase(),
    name: (entry.agent_name as string) ?? "Agent",
    model: (entry.model_used as string) ?? "",
    status: "done",
    content: (entry.content as string) ?? "",
    cost: typeof entry.cost === "number" ? entry.cost : 0,
    tokens: typeof entry.tokens === "number" ? entry.tokens : 0,
    badge: getAgentBadge(entry.agent_role, entry.entry_type),
  }));
}

export function LiveView({ sessionId, onReplay }: LiveViewProps) {
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [currentStep, setCurrentStep] = useState<string>("");
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  const [usedFallbackSynthesis, setUsedFallbackSynthesis] = useState(false);
  const [sessionStatus, setSessionStatus] = useState<
    "loading" | "running" | "complete" | "error" | "cancelled"
  >("loading");
  const [error, setError] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);
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
        setCurrentStep(normalizeStepId(data.step, data.name));
      } else if (type === "debate_start") {
        setCurrentStep("debate_a");
      } else if (type === "reconciliation_start") {
        setCompletedSteps((prev) => {
          const next = new Set(prev);
          next.add("debate_a");
          next.add("debate_b");
          return Array.from(next);
        });
        setCurrentStep("reconcile");
      } else if (type === "agent_start") {
        const agentId = data.agent ?? data.agent_id ?? `agent-${Date.now()}`;
        const role = (data.role ?? "reasoner").toLowerCase();
        const newAgent: AgentState = {
          id: agentId,
          role,
          name: data.agent ?? data.name ?? data.role ?? "Agent",
          model: data.model ?? "",
          status: "thinking",
          content: "",
          cost: 0,
          tokens: 0,
          badge: data.badge ?? getAgentBadge(role),
        };
        setAgents((prev) => {
          const existingIndex = prev.findIndex((agent) => agent.id === agentId);
          if (existingIndex >= 0) {
            return prev.map((agent) =>
              agent.id === agentId ? { ...agent, ...newAgent, content: agent.content } : agent,
            );
          }
          return [...prev, newAgent];
        });
        if (sessionInfo?.pattern === "sequential") {
          const stepId = SEQUENTIAL_AGENT_STEP_IDS[agentId];
          if (stepId) setCurrentStep(stepId);
        } else if (sessionInfo?.pattern === "debate") {
          const stepId = DEBATE_AGENT_STEP_IDS[agentId];
          if (stepId) setCurrentStep(stepId);
        }
      } else if (type === "agent_chunk") {
        const agentId = data.agent ?? data.agent_id;
        const chunk: string = data.content ?? data.chunk ?? "";
        setAgents((prev) =>
          prev.map((a) =>
            a.id === agentId ? { ...a, content: a.content + chunk } : a,
          ),
        );
      } else if (type === "agent_done") {
        const agentId = data.agent ?? data.agent_id;
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
        if (sessionInfo?.pattern === "debate") {
          const stepId = DEBATE_AGENT_STEP_IDS[agentId];
          if (stepId) {
            setCompletedSteps((prev) => (prev.includes(stepId) ? prev : [...prev, stepId]));
            setCurrentStep("");
          }
        } else if (sessionInfo?.pattern === "sequential") {
          const stepId = SEQUENTIAL_AGENT_STEP_IDS[agentId];
          if (stepId) {
            setCompletedSteps((prev) => (prev.includes(stepId) ? prev : [...prev, stepId]));
            setCurrentStep("");
          }
        }
      } else if (type === "step_complete") {
        const step = normalizeStepId(data.step, data.name);
        setCompletedSteps((prev) => (prev.includes(step) ? prev : [...prev, step]));
        setCurrentStep((prev) => (prev === step ? "" : prev));
      } else if (type === "session_complete") {
        setFinalAnswer(data.answer ?? data.result ?? null);
        setUsedFallbackSynthesis(Boolean(data.fallback_synthesis));
        setSessionStatus("complete");
        setCurrentStep("");
        setCompletedSteps((prev) => {
          const next = new Set(prev);
          if (sessionInfo?.pattern === "debate") {
            next.add("debate_a");
            next.add("debate_b");
            next.add("rebuttal");
            next.add("reconcile");
          } else if (sessionInfo?.pattern === "peer_review") {
            next.add("produce");
            next.add("review");
            next.add("rebuttal");
            next.add("react");
            next.add("synthesize");
          }
          return Array.from(next);
        });
        // Close SSE
        eventSourceRef.current?.close();
      } else if (type === "agent_error") {
        const agentId = data.agent ?? data.agent_id;
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
  }, [sessionInfo?.pattern]);

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
          setCompletedSteps(getPatternCompletedSteps(session.pattern ?? "sequential"));
          const scratchpad = (session as Record<string, unknown>).scratchpad_data as
            | { entries?: Array<Record<string, unknown>> }
            | undefined;
          const hasSynthesis = Boolean(
            scratchpad?.entries?.some((entry) => entry.entry_type === "synthesis"),
          );
          setUsedFallbackSynthesis(!hasSynthesis && Boolean(session.result ?? session.answer));
          // Reconstruct agents from session data if available
          setAgents(buildAgentsFromScratchpad(session as Record<string, unknown>));
          return;
        }
        if (session.status === "cancelled") {
          setSessionStatus("cancelled");
          setAgents(buildAgentsFromScratchpad(session as Record<string, unknown>));
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
                  sessionStatus === "cancelled" && "text-muted-foreground",
                  sessionStatus === "error" && "text-destructive",
                )}
              >
                {sessionStatus === "running" && "Running…"}
                {sessionStatus === "complete" && "Complete"}
                {sessionStatus === "cancelled" && "Stopped"}
                {sessionStatus === "error" && "Error"}
              </span>
              {derivedTotalCost > 0 && (
                <span className="text-xs font-mono text-muted-foreground">
                  ${derivedTotalCost.toFixed(4)} total
                </span>
              )}
            </div>
            {sessionInfo?.pattern && PATTERN_DESCRIPTIONS[sessionInfo.pattern] && (
              <p className="mt-2 text-sm text-muted-foreground">
                {PATTERN_DESCRIPTIONS[sessionInfo.pattern]}
              </p>
            )}
          </div>

          {sessionStatus === "running" ? (
            <Button
              variant="outline"
              size="sm"
              onClick={async () => {
                setIsStopping(true);
                try {
                  await stopSession(sessionId);
                  eventSourceRef.current?.close();
                  setSessionStatus("cancelled");
                  setCurrentStep("");
                  setError(null);
                } catch (stopError) {
                  setError(stopError instanceof Error ? stopError.message : "Failed to stop session.");
                } finally {
                  setIsStopping(false);
                }
              }}
              disabled={isStopping}
              className="shrink-0"
            >
              {isStopping ? "Stopping…" : "Stop Session"}
            </Button>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={onReplay}
              className="shrink-0"
            >
              Replay with Changes
            </Button>
          )}
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

      {sessionStatus === "cancelled" && !error && (
        <div className="rounded-lg border border-border bg-muted/30 px-4 py-3">
          <p className="text-sm font-medium text-foreground">Session stopped</p>
          <p className="text-sm text-muted-foreground mt-0.5">
            The discussion was cancelled before completion.
          </p>
        </div>
      )}

      {/* Final answer */}
      {finalAnswer && (
        <div className="rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-4">
          <p className="text-xs font-semibold text-emerald-700 uppercase tracking-wide mb-2">
            {usedFallbackSynthesis ? "Fallback Summary" : "Final Answer"}
          </p>
          {usedFallbackSynthesis && (
            <p className="mb-3 text-sm text-emerald-800/80">
              A full synthesizer output was not produced, so this summary was assembled from the agent outputs that did finish.
            </p>
          )}
          <MessageContent
            content={finalAnswer}
            className="text-sm leading-relaxed text-foreground"
          />
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
