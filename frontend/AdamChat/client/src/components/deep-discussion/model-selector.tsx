import { cn } from "@/lib/utils";

interface ModelSelectorProps {
  value: string;
  onChange: (modelId: string) => void;
  className?: string;
}

const MODEL_GROUPS = [
  {
    label: "X.AI",
    models: [
      { id: "grok-4.20-multi-agent-0309", name: "Grok Multi-Agent" },
      { id: "grok-4.20-0309-reasoning", name: "Grok Reasoning" },
      { id: "grok-4.20-0309-non-reasoning", name: "Grok Standard" },
    ],
  },
  {
    label: "Anthropic",
    models: [
      { id: "claude-opus-4-6", name: "Claude Opus" },
      { id: "claude-sonnet-4-6", name: "Claude Sonnet" },
      { id: "claude-haiku-4-5", name: "Claude Haiku" },
    ],
  },
  {
    label: "OpenAI",
    models: [
      { id: "gpt-5.4-2026-03-05", name: "GPT-5.4" },
      { id: "gpt-5.4-mini-2026-03-17", name: "GPT-5.4 Mini" },
    ],
  },
  {
    label: "Google",
    models: [
      { id: "gemini-3.1-pro-preview", name: "Gemini Pro" },
      { id: "gemini-3-flash-preview", name: "Gemini Flash" },
    ],
  },
];

export function ModelSelector({ value, onChange, className }: ModelSelectorProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(
        "w-full rounded-md border border-input bg-background px-3 py-2 text-sm",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-1",
        "cursor-pointer",
        className,
      )}
    >
      {MODEL_GROUPS.map((group) => (
        <optgroup key={group.label} label={group.label}>
          {group.models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}
