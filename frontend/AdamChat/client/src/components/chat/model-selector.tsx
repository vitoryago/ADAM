import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Brain, Zap, Sparkles, Bot } from "lucide-react";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const models = [
  {
    value: "automatic",
    name: "Automatic",
    description: "Let ADAM choose the best model",
    icon: Brain,
    color: "text-purple-600",
  },
  {
    value: "grok-4.20-multi-agent-0309",
    name: "Grok 4.20 Multi-Agent",
    description: "Best for multi-agent collaboration",
    icon: Sparkles,
    color: "text-blue-600",
  },
  {
    value: "grok-4.20-0309-reasoning",
    name: "Grok 4.20 Reasoning",
    description: "High complexity reasoning tasks",
    icon: Sparkles,
    color: "text-blue-500",
  },
  {
    value: "grok-4.20-0309-non-reasoning",
    name: "Grok 4.20 Standard",
    description: "Low/medium complexity queries (default)",
    icon: Zap,
    color: "text-green-600",
  },
  {
    value: "claude-opus-4-6",
    name: "Claude Opus 4.6",
    description: "Best for code generation",
    icon: Sparkles,
    color: "text-violet-600",
  },
  {
    value: "claude-sonnet-4-6",
    name: "Claude Sonnet 4.6",
    description: "Fast, excellent at synthesis",
    icon: Zap,
    color: "text-violet-500",
  },
  {
    value: "gpt-5.4-2026-03-05",
    name: "GPT-5.4",
    description: "OpenAI's flagship model",
    icon: Bot,
    color: "text-cyan-600",
  },
  {
    value: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro",
    description: "Google's analysis model",
    icon: Bot,
    color: "text-emerald-600",
  },
];

export function ModelSelector({ value, onChange, disabled }: ModelSelectorProps) {
  const selectedModel = models.find((m) => m.value === value) || models[0];

  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="w-[280px]">
        <SelectValue>
          <div className="flex items-center gap-2">
            <selectedModel.icon className={`w-4 h-4 ${selectedModel.color}`} />
            <span className="text-sm">{selectedModel.name}</span>
          </div>
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.value} value={model.value}>
            <div className="flex items-start gap-3">
              <model.icon className={`w-4 h-4 mt-0.5 ${model.color}`} />
              <div className="flex flex-col">
                <span className="font-medium">{model.name}</span>
                <span className="text-xs text-muted-foreground">{model.description}</span>
              </div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}