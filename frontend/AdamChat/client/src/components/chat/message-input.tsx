import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Paperclip, X, FileImage, FileCode, FileText, Mic, MicOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { CodeDemoButton } from "./code-demo-button";
import { VoiceInput } from "./voice-input";
import { StreamingVoiceConversation } from "./streaming-voice-conversation";

interface AttachedFile {
  name: string;
  type: string;
  size: number;
  data: string; // base64
  preview?: string; // for images
}

interface MessageInputProps {
  onSendMessage: (content: string, attachedFile?: AttachedFile) => void;
  disabled?: boolean;
  conversationId?: string;
  model?: string;
  useSearch?: boolean;
}

export function MessageInput({ 
  onSendMessage, 
  disabled = false,
  conversationId,
  model,
  useSearch = false
}: MessageInputProps) {
  const [message, setMessage] = useState("");
  const [attachedFile, setAttachedFile] = useState<AttachedFile | null>(null);
  const [voiceMode, setVoiceMode] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    const trimmedMessage = message.trim();
    if ((trimmedMessage || attachedFile) && !disabled) {
      onSendMessage(trimmedMessage, attachedFile || undefined);
      setMessage("");
      setAttachedFile(null);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Check file size (20MB limit)
    if (file.size > 20 * 1024 * 1024) {
      alert("File size must be less than 20MB");
      return;
    }

    // Read file as base64
    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64 = event.target?.result as string;
      const base64Data = base64.split(',')[1]; // Remove data:type;base64, prefix

      const fileData: AttachedFile = {
        name: file.name,
        type: file.type,
        size: file.size,
        data: base64Data
      };

      // If it's an image, create a preview
      if (file.type.startsWith('image/')) {
        fileData.preview = base64;
      }

      setAttachedFile(fileData);
    };
    reader.readAsDataURL(file);

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeFile = () => {
    setAttachedFile(null);
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return FileImage;
    if (type.includes('python') || type.includes('javascript') || type.includes('code')) return FileCode;
    return FileText;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 128) + "px";
    }
  }, [message]);

  const characterCount = message.length;
  const maxCharacters = 100000; // Increased from 4000 to 100k characters
  const isOverLimit = characterCount > maxCharacters;
  const canSend = (message.trim().length > 0 || attachedFile) && !disabled && !isOverLimit;

  return (
    <div className="border-t border-border bg-background p-4">
      <div className="max-w-4xl mx-auto">
        {/* Voice Mode Toggle */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Button
              variant={voiceMode ? "default" : "outline"}
              size="sm"
              onClick={() => setVoiceMode(!voiceMode)}
              disabled={disabled || !conversationId}
              className="h-8"
            >
              {voiceMode ? (
                <>
                  <MicOff className="w-4 h-4 mr-2" />
                  Exit Voice Mode
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4 mr-2" />
                  Voice Mode
                </>
              )}
            </Button>
            {voiceMode && (
              <span className="text-sm text-muted-foreground">
                Click the mic button below to start talking
              </span>
            )}
          </div>
        </div>

        {voiceMode && conversationId ? (
          <StreamingVoiceConversation
            conversationId={conversationId}
            className="mb-4"
          />
        ) : (
        <div className="relative">
          {/* File Preview */}
          {attachedFile && (
            <div className="mb-3 p-3 bg-muted rounded-lg flex items-center justify-between">
              <div className="flex items-center gap-3">
                {attachedFile.preview ? (
                  <img src={attachedFile.preview} alt={attachedFile.name} className="w-12 h-12 object-cover rounded" />
                ) : (
                  <div className="w-12 h-12 bg-background rounded flex items-center justify-center">
                    {(() => {
                      const Icon = getFileIcon(attachedFile.type);
                      return <Icon className="w-6 h-6 text-muted-foreground" />;
                    })()}
                  </div>
                )}
                <div className="flex flex-col">
                  <span className="text-sm font-medium">{attachedFile.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {(attachedFile.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={removeFile}
                className="h-8 w-8"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          )}

          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <Textarea
                ref={textareaRef}
                placeholder="Message ADAM..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={disabled}
                className={cn(
                  "min-h-[3rem] max-h-32 px-4 py-3 pr-12 bg-muted border-border rounded-2xl resize-none",
                  "focus:ring-2 focus:ring-adam-primary focus:border-transparent",
                  "placeholder:text-muted-foreground",
                  isOverLimit && "border-destructive focus:ring-destructive"
                )}
                rows={1}
              />
              <Button
                onClick={handleSubmit}
                disabled={!canSend}
                size="icon"
                className={cn(
                  "absolute right-2 bottom-2 w-8 h-8 rounded-full",
                  "bg-adam-primary hover:bg-adam-primary/90",
                  "disabled:bg-muted-foreground disabled:cursor-not-allowed"
                )}
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-3">
            <div className="flex items-center gap-2">
              <CodeDemoButton onSendMessage={onSendMessage} />
              <VoiceInput 
                onTranscription={(text) => {
                  setMessage(text);
                }}
                disabled={disabled}
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                className="h-8 px-3"
              >
                <Paperclip className="w-4 h-4 mr-2" />
                Attach File
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                accept="image/*,.py,.js,.jsx,.ts,.tsx,.json,.txt,.md,.csv,.html,.css,.scss,.yaml,.yml,.xml,.sql,.sh,.bash,.zsh,.cpp,.c,.h,.hpp,.java,.rs,.go,.rb,.php,.swift"
                className="hidden"
              />
              <p className="text-xs text-muted-foreground">
                ADAM can make mistakes. Consider checking important information.
              </p>
            </div>
            <div className={cn(
              "flex items-center space-x-2 text-xs",
              isOverLimit ? "text-destructive" : "text-muted-foreground"
            )}>
              <span>{characterCount}</span>
              <span>/</span>
              <span>{maxCharacters}</span>
            </div>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}
