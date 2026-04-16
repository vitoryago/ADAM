import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { BookOpen, Briefcase, MessageSquare, Scale, Sparkles } from "lucide-react";

interface ResponseStyle {
  value: string;
  label: string;
  description: string;
  icon?: React.ReactNode;
}

const responseStyles: ResponseStyle[] = [
  {
    value: "concise",
    label: "Concise",
    description: "Brief, to-the-point answers",
    icon: <MessageSquare className="w-4 h-4" />
  },
  {
    value: "normal",
    label: "Normal",
    description: "Balanced default responses",
    icon: <Scale className="w-4 h-4" />
  },
  {
    value: "explanatory",
    label: "Explanatory",
    description: "Detailed explanations with examples",
    icon: <BookOpen className="w-4 h-4" />
  },
  {
    value: "formal",
    label: "Formal",
    description: "Professional, formal tone",
    icon: <Briefcase className="w-4 h-4" />
  },
  {
    value: "friendly",
    label: "Friendly",
    description: "Casual, conversational tone",
    icon: <Sparkles className="w-4 h-4" />
  },
  {
    value: "educational",
    label: "Educational",
    description: "Teaching style with step-by-step",
    icon: <BookOpen className="w-4 h-4" />
  },
  {
    value: "creative",
    label: "Creative",
    description: "Engaging, creative responses",
    icon: <Sparkles className="w-4 h-4" />
  }
];

interface ResponseStyleSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
  className?: string;
}

export function ResponseStyleSelector({ 
  value, 
  onValueChange,
  className 
}: ResponseStyleSelectorProps) {
  // Load saved preference from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("adam-response-style");
    if (saved && responseStyles.find(s => s.value === saved)) {
      onValueChange(saved);
    } else {
      // Default to "normal"
      onValueChange("normal");
      localStorage.setItem("adam-response-style", "normal");
    }
  }, []);

  const handleChange = (newValue: string) => {
    onValueChange(newValue);
    localStorage.setItem("adam-response-style", newValue);
  };

  const currentStyle = responseStyles.find(s => s.value === value);

  return (
    <Select value={value} onValueChange={handleChange}>
      <SelectTrigger className={`w-[140px] ${className}`}>
        <SelectValue>
          <div className="flex items-center gap-1.5">
            {currentStyle?.icon}
            <span className="text-sm">{currentStyle?.label || "Normal"}</span>
          </div>
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>Response Style</SelectLabel>
          {responseStyles.map((style) => (
            <SelectItem key={style.value} value={style.value}>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  {style.icon}
                  <span className="font-medium">{style.label}</span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {style.description}
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
