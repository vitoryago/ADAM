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
    value: "claude-opus-4.1",
    name: "Claude Opus 4.1",
    description: "Most powerful with deep thinking",
    icon: Sparkles,
    color: "text-violet-600",
  },
  {
    value: "claude-3.5-sonnet",
    name: "Claude 3.5 Sonnet",
    description: "Fast & capable with vision",
    icon: Bot,
    color: "text-indigo-600",
  },
  {
    value: "claude-3.5-haiku",
    name: "Claude 3.5 Haiku",
    description: "Lightning fast responses",
    icon: Zap,
    color: "text-purple-600",
  },
  {
    value: "gpt-5",
    name: "GPT-5",
    description: "Most capable with vision support",
    icon: Sparkles,
    color: "text-red-600",
  },
  {
    value: "gpt-5-mini",
    name: "GPT-5 Mini",
    description: "Fast and efficient",
    icon: Zap,
    color: "text-orange-600",
  },
  {
    value: "gpt-5-nano",
    name: "GPT-5 Nano",
    description: "Ultra-fast for simple queries",
    icon: Zap,
    color: "text-yellow-600",
  },
  {
    value: "grok-4-reasoning",
    name: "Grok 4 Reasoning",
    description: "Deep reasoning for complex tasks",
    icon: Sparkles,
    color: "text-blue-600",
  },
  {
    value: "grok-4",
    name: "Grok 4",
    description: "Standard high-quality responses",
    icon: Bot,
    color: "text-green-600",
  },
  {
    value: "grok-2-vision-1212",
    name: "Grok 2 Vision",
    description: "Optimized for image analysis",
    icon: Bot,
    color: "text-sky-600",
  },
  {
    value: "grok-3-mini-high",
    name: "Grok 3 Mini",
    description: "Balanced speed with high quality",
    icon: Zap,
    color: "text-teal-600",
  },
  {
    value: "o4-mini-high",
    name: "O4 Mini",
    description: "OpenAI reasoning model",
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