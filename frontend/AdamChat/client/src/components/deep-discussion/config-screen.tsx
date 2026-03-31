import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { PatternSelector } from "./pattern-selector";
import { ModelSelector } from "./model-selector";
import { createSession, updateSessionConfig } from "@/lib/deep-discussion-api";

interface AgentCard {
  role: string;
  badge: "Producer" | "Reviewer" | "Final";
  description: string;
  defaultModelKey: string;
}

const AGENT_CARDS: AgentCard[] = [
  {
    role: "Planner",
    badge: "Producer",
    description: "Breaks down the problem into structured steps and produces an initial plan.",
    defaultModelKey: "planner",
  },
  {
    role: "Implementer",
    badge: "Producer",
    description: "Executes the plan or codes the solution based on the planner's output.",
    defaultModelKey: "implementer",
  },
  {
    role: "Reviewer",
    badge: "Reviewer",
    description: "Critiques the implementation for correctness, quality, and edge cases.",
    defaultModelKey: "reviewer",
  },
  {
    role: "Synthesizer",
    badge: "Final",
    description: "Integrates all feedback and produces the final polished response.",
    defaultModelKey: "synthesizer",
  },
];

const BADGE_STYLES: Record<string, string> = {
  Producer: "bg-blue-100 text-blue-700",
  Reviewer: "bg-amber-100 text-amber-700",
  Final: "bg-emerald-100 text-emerald-700",
};

const DEFAULT_MODEL = "claude-sonnet-4-6";
const DEFAULT_BUDGET = 2.0;

interface ConfigScreenProps {
  projectId: string;
  onStart: (sessionId: string) => void;
  conversationId?: string;
}

export function ConfigScreen({ projectId, onStart, conversationId }: ConfigScreenProps) {
  const [question, setQuestion] = useState("");
  const [pattern, setPattern] = useState("sequential");
  const [modelAssignments, setModelAssignments] = useState<Record<string, string>>(
    Object.fromEntries(AGENT_CARDS.map((a) => [a.defaultModelKey, DEFAULT_MODEL])),
  );
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const [preferLocal, setPreferLocal] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const questionRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    questionRef.current?.focus();
  }, []);

  const defaultAssignments = Object.fromEntries(
    AGENT_CARDS.map((a) => [a.defaultModelKey, DEFAULT_MODEL]),
  );

  const hasCustomAssignments =
    JSON.stringify(modelAssignments) !== JSON.stringify(defaultAssignments);

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
        preferLocal,
      );
      if (hasCustomAssignments || budget !== DEFAULT_BUDGET) {
        await updateSessionConfig(session.id, {
          model_assignments: modelAssignments,
          budget,
        });
      }
      return session;
    },
    onSuccess: (session) => {
      onStart(session.id);
    },
    onError: (err: Error) => {
      setError(err.message);
    },
  });

  const handleModelChange = (agentKey: string, modelId: string) => {
    setModelAssignments((prev) => ({ ...prev, [agentKey]: modelId }));
  };

  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto py-8 px-4">
      <div>
        <h1 className="text-2xl font-bold mb-1">Deep Discussion</h1>
        <p className="text-muted-foreground text-sm">
          Assign multiple AI agents to reason through a question collaboratively.
        </p>
      </div>

      {/* Question */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold">Your Question</label>
        <Textarea
          ref={questionRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What would you like the agents to discuss in depth?"
          className="min-h-[100px] resize-none"
        />
      </div>

      {/* Pattern */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold">Discussion Pattern</label>
        <PatternSelector value={pattern} onChange={setPattern} />
      </div>

      {/* Prefer Local Toggle */}
      <div className="flex items-center justify-between py-2">
        <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <span>Prefer Local Models</span>
        </label>
        <button
          type="button"
          role="switch"
          aria-checked={preferLocal}
          onClick={() => setPreferLocal(!preferLocal)}
          className={cn(
            "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
            preferLocal ? "bg-primary" : "bg-muted",
          )}
        >
          <span
            className={cn(
              "inline-block h-3.5 w-3.5 transform rounded-full bg-background transition-transform",
              preferLocal ? "translate-x-4" : "translate-x-0.5",
            )}
          />
        </button>
      </div>

      {/* Agent Cards */}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold">Agent Models</label>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {AGENT_CARDS.map((agent) => (
            <div
              key={agent.defaultModelKey}
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
                value={modelAssignments[agent.defaultModelKey]}
                onChange={(modelId) => handleModelChange(agent.defaultModelKey, modelId)}
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

      {/* Submit */}
      <Button
        onClick={() => {
          setError(null);
          startMutation.mutate();
        }}
        disabled={startMutation.isPending || !question.trim()}
        className="w-full bg-primary text-primary-foreground hover:bg-primary/90"
        size="lg"
      >
        {startMutation.isPending ? "Starting…" : "Start Deep Discussion"}
      </Button>
    </div>
  );
}
