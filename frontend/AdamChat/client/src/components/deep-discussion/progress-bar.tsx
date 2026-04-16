import { cn } from "@/lib/utils";

interface ProgressBarProps {
  pattern: string;
  currentStep: string;
  completedSteps: string[];
}

const PATTERN_STEPS: Record<string, { id: string; label: string }[]> = {
  sequential: [
    { id: "produce", label: "Produce" },
    { id: "code", label: "Code" },
    { id: "review", label: "Review" },
    { id: "synthesize", label: "Synthesize" },
  ],
  debate: [
    { id: "debate_a", label: "Position A" },
    { id: "debate_b", label: "Position B" },
    { id: "rebuttal", label: "Rebuttal" },
    { id: "reconcile", label: "Reconcile" },
  ],
  peer_review: [
    { id: "produce", label: "Produce" },
    { id: "review", label: "Review" },
    { id: "rebuttal", label: "Rebuttal" },
    { id: "react", label: "React" },
    { id: "synthesize", label: "Synthesize" },
  ],
};

export function ProgressBar({ pattern, currentStep, completedSteps }: ProgressBarProps) {
  const steps = PATTERN_STEPS[pattern] ?? PATTERN_STEPS["sequential"];

  return (
    <div className="w-full">
      <div className="flex gap-1">
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = currentStep === step.id && !isCompleted;
          const isPending = !isCompleted && !isCurrent;

          return (
            <div key={step.id} className="flex-1 flex flex-col gap-1">
              <div
                className={cn(
                  "h-2.5 rounded-full transition-all duration-500",
                  isCompleted && "bg-emerald-500",
                  isCurrent && "bg-amber-400 animate-pulse",
                  isPending && "bg-muted",
                )}
              />
              <span
                className={cn(
                  "text-xs text-center truncate",
                  isCompleted && "text-emerald-600 font-medium",
                  isCurrent && "text-amber-600 font-semibold",
                  isPending && "text-muted-foreground",
                )}
              >
                {index + 1}. {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
