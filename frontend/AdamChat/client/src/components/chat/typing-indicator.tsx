import { Brain } from "lucide-react";

export function TypingIndicator() {
  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex items-start space-x-3 max-w-2xl">
        <div className="w-8 h-8 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0">
          <Brain className="w-4 h-4 text-white" />
        </div>
        <div className="bg-muted rounded-2xl rounded-tl-md px-4 py-3 shadow-sm">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-pulse-dot" />
            <div 
              className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-pulse-dot" 
              style={{ animationDelay: "0.2s" }}
            />
            <div 
              className="w-2 h-2 bg-muted-foreground/60 rounded-full animate-pulse-dot" 
              style={{ animationDelay: "0.4s" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
