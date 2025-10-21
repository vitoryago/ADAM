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
    value: "grok-4-fast-reasoning",
    name: "Grok 4 Fast Reasoning",
    description: "High complexity reasoning tasks",
    icon: Sparkles,
    color: "text-blue-600",
  },
  {
    value: "grok-4-fast-non-reasoning",
    name: "Grok 4 Fast",
    description: "Low/medium complexity queries (default)",
    icon: Zap,
    color: "text-green-600",
  },
  {
    value: "claude-sonnet-4-5",
    name: "Claude Sonnet 4.5",
    description: "Best for code generation",
    icon: Sparkles,
    color: "text-violet-600",
  },
  {
    value: "grok-2-vision-1212",
    name: "Grok 2 Vision",
    description: "Optimized for image analysis",
    icon: Bot,
    color: "text-indigo-600",
  },
  {
    value: "grok-4-reasoning",
    name: "Grok 4 Reasoning (Legacy)",
    description: "Older reasoning model",
    icon: Bot,
    color: "text-blue-500",
  },
  {
    value: "grok-4",
    name: "Grok 4 (Legacy)",
    description: "Older standard model",
    icon: Bot,
    color: "text-green-500",
  },
  {
    value: "gpt-4",
    name: "GPT-4",
    description: "OpenAI's flagship model",
    icon: Bot,
    color: "text-cyan-600",
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