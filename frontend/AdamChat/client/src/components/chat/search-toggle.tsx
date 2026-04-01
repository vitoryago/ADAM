import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Globe, Twitter } from "lucide-react";
import { cn } from "@/lib/utils";

interface SearchToggleProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  searchMode: string;
  onSearchModeChange: (mode: string) => void;
  disabled?: boolean;
}

const searchModes = [
  {
    value: "web",
    name: "Web",
    description: "Search the entire web",
    icon: Globe,
  },
  {
    value: "x",
    name: "X/Twitter",
    description: "Search X (Twitter) posts",
    icon: Twitter,
  },
];

export function SearchToggle({ 
  enabled, 
  onEnabledChange, 
  searchMode,
  onSearchModeChange,
  disabled 
}: SearchToggleProps) {
  const selectedMode = searchModes.find((m) => m.value === searchMode) || searchModes[0];

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Switch
          id="search-toggle"
          checked={enabled}
          onCheckedChange={onEnabledChange}
          disabled={disabled}
        />
        <Label 
          htmlFor="search-toggle" 
          className={cn(
            "text-sm cursor-pointer",
            disabled && "cursor-not-allowed opacity-50"
          )}
        >
          <div className="flex items-center gap-1.5">
            <Search className="w-3.5 h-3.5" />
            <span>Live Search</span>
          </div>
        </Label>
      </div>

      {enabled && (
        <Select 
          value={searchMode} 
          onValueChange={onSearchModeChange} 
          disabled={disabled}
        >
          <SelectTrigger className="w-[140px]">
            <SelectValue>
              <div className="flex items-center gap-2">
                <selectedMode.icon className="w-3.5 h-3.5" />
                <span className="text-sm">{selectedMode.name}</span>
              </div>
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {searchModes.map((mode) => (
              <SelectItem key={mode.value} value={mode.value}>
                <div className="flex items-start gap-2">
                  <mode.icon className="w-3.5 h-3.5 mt-0.5" />
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">{mode.name}</span>
                    <span className="text-xs text-muted-foreground">{mode.description}</span>
                  </div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}