import { useState } from "react";
import { cn } from "@/lib/utils";
import { MessageContent } from "@/lib/message-parser";

export interface AgentCardProps {
  role: string;
  name: string;
  model: string;
  status: "pending" | "thinking" | "done" | "error";
  content: string;
  cost: number;
  tokens: number;
  badge: string;
}

const ROLE_DOT_COLORS: Record<string, string> = {
  reasoner: "bg-blue-500",
  coder: "bg-emerald-500",
  critic: "bg-orange-500",
  synthesizer: "bg-purple-500",
};

const BADGE_STYLES: Record<string, string> = {
  PRODUCER: "bg-blue-100 text-blue-700",
  REVIEWER: "bg-amber-100 text-amber-700",
  REBUTTAL: "bg-orange-100 text-orange-700",
  FINAL: "bg-emerald-100 text-emerald-700",
};

const STATUS_INDICATORS: Record<string, { label: string; className: string }> = {
  pending: { label: "Waiting", className: "text-muted-foreground" },
  thinking: { label: "Thinking…", className: "text-amber-600" },
  done: { label: "Done", className: "text-emerald-600" },
  error: { label: "Error", className: "text-destructive" },
};

export function AgentCard({
  role,
  name,
  model,
  status,
  content,
  cost,
  tokens,
  badge,
}: AgentCardProps) {
  const [expanded, setExpanded] = useState(true);

  const dotColor = ROLE_DOT_COLORS[role.toLowerCase()] ?? "bg-gray-400";
  const badgeStyle = BADGE_STYLES[badge?.toUpperCase()] ?? "bg-muted text-muted-foreground";
  const statusInfo = STATUS_INDICATORS[status];

  const modelShort = model.split("/").pop() ?? model;

  return (
    <div
      className={cn(
        "rounded-lg border bg-card transition-all duration-300",
        status === "pending" && "border-dashed border-muted-foreground/40",
        status === "thinking" && "border-amber-300",
        status === "done" && "border-border",
        status === "error" && "border-destructive/50",
      )}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 px-4 py-3 cursor-pointer select-none"
        onClick={() => setExpanded((v) => !v)}
      >
        {/* Role dot */}
        <span className={cn("w-2.5 h-2.5 rounded-full shrink-0", dotColor)} />

        {/* Name */}
        <span className="font-semibold text-sm">{name}</span>

        {/* Badge */}
        {badge && (
          <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", badgeStyle)}>
            {badge}
          </span>
        )}

        {/* Spacer */}
        <span className="flex-1" />

        {/* Model */}
        <span className="text-xs text-muted-foreground hidden sm:inline">{modelShort}</span>

        {/* Cost badge */}
        {cost > 0 && (
          <span className="text-xs rounded-full bg-muted px-2 py-0.5 font-mono">
            ${cost.toFixed(4)}
          </span>
        )}

        {/* Tokens */}
        {tokens > 0 && (
          <span className="text-xs text-muted-foreground font-mono">{tokens.toLocaleString()} tok</span>
        )}

        {/* Status indicator */}
        <div className="flex items-center gap-1.5">
          {status === "thinking" && (
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          )}
          <span className={cn("text-xs font-medium", statusInfo.className)}>
            {statusInfo.label}
          </span>
        </div>

        {/* Expand/collapse chevron */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={cn(
            "text-muted-foreground transition-transform duration-200 shrink-0",
            expanded ? "rotate-180" : "rotate-0",
          )}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {/* Content */}
      {expanded && (
        <div className="px-4 pb-4">
          {status === "pending" && !content && (
            <p className="text-sm text-muted-foreground italic">Waiting to start…</p>
          )}

          {status === "thinking" && !content && (
            <div className="flex items-center gap-2 text-sm text-amber-600">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <span>Thinking…</span>
            </div>
          )}

          {status === "error" && !content && (
            <p className="text-sm text-destructive">An error occurred during processing.</p>
          )}

          {content && (
            <div className="relative">
              <MessageContent
                content={content}
                className="text-sm leading-relaxed text-foreground"
              />
              {status === "thinking" && (
                <span className="inline-flex items-center gap-1 text-amber-600 text-xs ml-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
