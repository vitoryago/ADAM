import { cn } from "@/lib/utils";

interface PatternSelectorProps {
  value: string;
  onChange: (pattern: string) => void;
}

const PATTERNS = [
  {
    id: "sequential",
    name: "Sequential",
    description: "Plan → Code → Review → Synthesize",
    steps: 4,
  },
  {
    id: "debate",
    name: "Debate",
    description: "Two perspectives → Reconcile",
    steps: 3,
  },
  {
    id: "peer_review",
    name: "Peer Review",
    description: "Produce → Review → Rebut → React → Synthesize",
    steps: 5,
  },
];

export function PatternSelector({ value, onChange }: PatternSelectorProps) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      {PATTERNS.map((pattern) => {
        const isSelected = value === pattern.id;
        return (
          <button
            key={pattern.id}
            type="button"
            onClick={() => onChange(pattern.id)}
            className={cn(
              "rounded-lg border-2 p-4 text-left transition-all",
              "hover:border-primary/60 hover:bg-primary/5",
              isSelected
                ? "border-primary bg-primary/10"
                : "border-border bg-background",
            )}
          >
            <div className="mb-1 font-semibold text-sm">{pattern.name}</div>
            <div className="text-xs text-muted-foreground mb-2">
              {pattern.description}
            </div>
            <div
              className={cn(
                "inline-block rounded-full px-2 py-0.5 text-xs font-medium",
                isSelected
                  ? "bg-primary/20 text-primary"
                  : "bg-muted text-muted-foreground",
              )}
            >
              {pattern.steps} steps
            </div>
          </button>
        );
      })}
    </div>
  );
}
