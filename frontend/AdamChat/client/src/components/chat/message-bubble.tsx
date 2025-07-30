import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Brain, User, Copy, Check, Cpu, Coins, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { MessageContent } from "@/lib/message-parser";
import type { Message } from "@shared/schema";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const { toast } = useToast();
  const [copied, setCopied] = useState(false);

  const isUser = message.role === "user";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Copied to clipboard",
        description: "Message copied successfully.",
      });
    } catch (error) {
      toast({
        title: "Copy failed",
        description: "Failed to copy message to clipboard.",
        variant: "destructive",
      });
    }
  };

  const formatTimestamp = (date: Date | string | null) => {
    if (!date) return "";
    const d = new Date(date);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };



  return (
    <div className={cn(
      "flex animate-slide-up",
      isUser ? "justify-end" : "justify-start"
    )}>
      <div className="flex items-start space-x-3 max-w-2xl">
        {!isUser && (
          <div className="w-8 h-8 bg-gradient-to-br from-adam-primary to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0">
            <Brain className="w-4 h-4 text-white" />
          </div>
        )}
        
        <div className={cn(
          "rounded-2xl px-4 py-3 shadow-sm relative group min-w-0 max-w-full",
          "inline-block fit-content",
          isUser ? (
            "bg-adam-primary text-white rounded-tr-md"
          ) : (
            "bg-muted rounded-tl-md"
          )
        )}>
          <div className={cn(
            "prose-chat text-sm leading-relaxed message-bubble-content",
            isUser ? "text-white" : "text-foreground"
          )}>
            {/* Display attached file if present */}
            {message.metadata?.preview && (
              <div className="mb-3">
                <img 
                  src={message.metadata.preview} 
                  alt={message.metadata.file_name || 'Attached image'} 
                  className="max-w-full rounded-lg"
                  style={{ maxHeight: '300px' }}
                />
              </div>
            )}
            {message.metadata?.file_name && !message.metadata.preview && (
              <div className="mb-3 p-2 bg-background/10 rounded flex items-center gap-2">
                <FileText className="w-4 h-4" />
                <span className="text-sm">{message.metadata.file_name}</span>
              </div>
            )}
            
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <MessageContent 
                content={message.content}
                className="prose prose-sm max-w-none dark:prose-invert"
              />
            )}
          </div>
          
          {isUser && message.metadata?.tokens_used && (
            <div className="flex items-center gap-1 mt-2 text-xs opacity-70">
              <Coins className="w-3 h-3" />
              <span>{message.metadata.tokens_used} tokens</span>
            </div>
          )}
          
          {!isUser && (
            <div className="flex flex-col gap-2 mt-3 pt-2 border-t border-border/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  {message.metadata?.model && (
                    <div className="flex items-center gap-1">
                      <Cpu className="w-3 h-3" />
                      <span>{message.metadata.model}</span>
                    </div>
                  )}
                  {message.metadata?.tokens_used && (
                    <div className="flex items-center gap-1">
                      <Coins className="w-3 h-3" />
                      <span>{message.metadata.tokens_used} tokens</span>
                    </div>
                  )}
                  <span>{formatTimestamp(message.timestamp)}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="opacity-0 group-hover:opacity-100 h-6 w-6"
                  onClick={handleCopy}
                  title="Copy message"
                >
                  {copied ? (
                    <Check className="w-3 h-3 text-green-500" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </Button>
              </div>
            </div>
          )}
        </div>
        
        {isUser && (
          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
            <User className="w-4 h-4 text-white" />
          </div>
        )}
      </div>
    </div>
  );
}
