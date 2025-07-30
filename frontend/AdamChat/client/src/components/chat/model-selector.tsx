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
    value: "grok-4-reasoning",
    name: "Grok 4 Reasoning",
    description: "Most powerful for complex tasks",
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
    color: "text-indigo-600",
  },
  {
    value: "grok-3-mini-high",
    name: "Grok 3 Mini (High)",
    description: "Fast responses with reasoning",
    icon: Zap,
    color: "text-yellow-600",
  },
  {
    value: "grok-3-mini-fast",
    name: "Grok 3 Mini (Fast)",
    description: "Fastest responses",
    icon: Zap,
    color: "text-orange-600",
  },
  {
    value: "gpt-4",
    name: "GPT-4",
    description: "OpenAI's flagship model",
    icon: Bot,
    color: "text-cyan-600",
  },
  {
    value: "gpt-3.5-turbo",
    name: "GPT-3.5 Turbo",
    description: "Fast and efficient",
    icon: Zap,
    color: "text-teal-600",
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