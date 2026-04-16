import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { PatternSelector } from "./pattern-selector";
import { ModelSelector } from "./model-selector";
import { createSession, updateSessionConfig } from "@/lib/deep-discussion-api";
import { X } from "lucide-react";

interface GoDeepModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectId: string;
  conversationId: string;
}

interface AgentCard {
  role: string;
  badge: "Producer" | "Reviewer" | "Final";
  description: string;
  assignmentKey: string;
}

const DEFAULT_AGENT_CARDS: AgentCard[] = [
  {
    role: "Reasoner",
    badge: "Producer",
    description: "Analyzes the question and frames the first structured approach.",
    assignmentKey: "reasoner",
  },
  {
    role: "Coder",
    badge: "Producer",
    description: "Executes the plan or builds the concrete implementation.",
    assignmentKey: "coder",
  },
  {
    role: "Critic",
    badge: "Reviewer",
    description: "Challenges the work for correctness, trade-offs, and edge cases.",
    assignmentKey: "critic",
  },
  {
    role: "Synthesizer",
    badge: "Final",
    description: "Integrates all feedback and produces the final polished response.",
    assignmentKey: "synthesizer",
  },
];

const DEBATE_AGENT_CARDS: AgentCard[] = [
  {
    role: "Perspective A",
    badge: "Producer",
    description: "Opens the debate and later returns for the rebuttal using the reasoner slot.",
    assignmentKey: "reasoner",
  },
  {
    role: "Perspective B",
    badge: "Producer",
    description: "Counters the opening argument and presses the opposing case using the coder slot.",
    assignmentKey: "coder",
  },
  {
    role: "Reconciler",
    badge: "Reviewer",
    description: "Weighs both arguments and produces the final reconciliation using the critic slot.",
    assignmentKey: "critic",
  },
];

const BADGE_STYLES: Record<string, string> = {
  Producer: "bg-blue-100 text-blue-700",
  Reviewer: "bg-amber-100 text-amber-700",
  Final: "bg-emerald-100 text-emerald-700",
};

const DEFAULT_MODEL = "claude-sonnet-4-6";
const DEFAULT_BUDGET = 2.0;

function getVisibleAgentCards(pattern: string): AgentCard[] {
  if (pattern === "debate") return DEBATE_AGENT_CARDS;
  return DEFAULT_AGENT_CARDS;
}

export function GoDeepModal({ isOpen, onClose, projectId, conversationId }: GoDeepModalProps) {
  const [, setLocation] = useLocation();
  const [question, setQuestion] = useState("");
  const [pattern, setPattern] = useState("sequential");
  const [modelAssignments, setModelAssignments] = useState<Record<string, string>>(
    Object.fromEntries(DEFAULT_AGENT_CARDS.map((a) => [a.assignmentKey, DEFAULT_MODEL])),
  );
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const [error, setError] = useState<string | null>(null);

  const visibleAgentCards = getVisibleAgentCards(pattern);

  const startMutation = useMutation({
    mutationFn: async () => {
      if (!question.trim()) {
        throw new Error("Please enter a question.");
      }
      const session = await createSession(
        projectId,
        question.trim(),
        pattern,
        conversationId,
      );
      await updateSessionConfig(session.id, {
        model_assignments: modelAssignments,
        budget,
      });
      return session;
    },
    onSuccess: (session) => {
      onClose();
      setLocation(`/project/${projectId}/deep-discussion/${session.id}`);
    },
    onError: (err: Error) => {
      setError(err.message);
    },
  });

  const handleModelChange = (agentKey: string, modelId: string) => {
    setModelAssignments((prev) => ({ ...prev, [agentKey]: modelId }));
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.6)" }}
    >
      <div
        className="relative bg-background rounded-2xl shadow-2xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border sticky top-0 bg-background rounded-t-2xl z-10">
          <div>
            <h2 className="text-xl font-bold">Deep Discussion</h2>
            <p className="text-sm text-muted-foreground">
              Escalate this conversation with multiple AI agents reasoning collaboratively.
            </p>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="shrink-0">
            <X className="w-5 h-5" />
          </Button>
        </div>

        {/* Body */}
        <div className="flex flex-col gap-6 px-6 py-6">
          {/* Question */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold">Your Question</label>
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What would you like the agents to discuss in depth?"
              className="min-h-[80px] resize-none"
              autoFocus
            />
          </div>

          {/* Pattern */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold">Discussion Pattern</label>
            <PatternSelector value={pattern} onChange={setPattern} />
          </div>

          {/* Agent Cards */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold">Agent Models</label>
            {pattern === "debate" && (
              <p className="text-xs text-muted-foreground">
                Debate runs as point, counterpoint, rebuttal, then reconciliation. Perspective A is used twice.
              </p>
            )}
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {visibleAgentCards.map((agent) => (
                <div
                  key={agent.assignmentKey}
                  className="rounded-lg border border-border bg-card p-4 flex flex-col gap-2"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-sm">{agent.role}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${BADGE_STYLES[agent.badge]}`}
                    >
                      {agent.badge}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">{agent.description}</p>
                  <ModelSelector
                    value={modelAssignments[agent.assignmentKey]}
                    onChange={(modelId) => handleModelChange(agent.assignmentKey, modelId)}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Budget */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold">
              Budget{" "}
              <span className="font-normal text-muted-foreground">
                (${budget.toFixed(2)})
              </span>
            </label>
            <input
              type="range"
              min={0.5}
              max={5.0}
              step={0.25}
              value={budget}
              onChange={(e) => setBudget(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>$0.50</span>
              <span>$5.00</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">
              {error}
            </p>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            <Button variant="outline" onClick={onClose} className="flex-1">
              Cancel
            </Button>
            <Button
              onClick={() => {
                setError(null);
                startMutation.mutate();
              }}
              disabled={startMutation.isPending || !question.trim()}
              className="flex-1 bg-emerald-600 text-white hover:bg-emerald-700"
            >
              {startMutation.isPending ? "Starting…" : "Start Deep Discussion"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
