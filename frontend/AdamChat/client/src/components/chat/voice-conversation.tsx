import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, MicOff, Loader2, Volume2, Code2, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { AudioPlayer } from "./audio-player";
import { CodeBlock } from "@/components/ui/code-highlighter";
import { useConversation } from "@/hooks/use-conversation";

interface VoiceConversationProps {
  conversationId: string;
  model?: string;
  useSearch?: boolean;
  className?: string;
}

interface VoiceResponse {
  spokenText: string;
  fullResponse: string;
  hasCode: boolean;
  codeBlocks?: Array<{
    language: string;
    code: string;
  }>;
  audioUrl?: string;
  waitForResponse: boolean;
}

export function VoiceConversation({ 
  conversationId, 
  model,
  useSearch = false,
  className 
}: VoiceConversationProps) {
  const { toast } = useToast();
  const { refetch } = useConversation(conversationId);
  
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResponse, setLastResponse] = useState<VoiceResponse | null>(null);
  const [showFullResponse, setShowFullResponse] = useState(false);
  const [conversationState, setConversationState] = useState<'idle' | 'listening' | 'processing' | 'speaking' | 'waiting'>('idle');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      // Detect silence for auto-stop
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      let silenceStart: number | null = null;
      let hasSpoken = false;
      const checkAudioLevel = () => {
        if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') return;
        
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        
        // Adjust threshold based on whether user has started speaking
        const threshold = hasSpoken ? 3 : 10;
        
        if (average > threshold) {
          hasSpoken = true;
          silenceStart = null;
        } else if (hasSpoken) { // Only detect silence after user has spoken
          if (!silenceStart) {
            silenceStart = Date.now();
          } else if (Date.now() - silenceStart > 1000) { // 1 second of silence
            stopRecording();
            return;
          }
        }
        
        requestAnimationFrame(checkAudioLevel);
      };

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await processVoiceChat(audioBlob);
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop());
        audioContext.close();
      };

      mediaRecorder.start(100); // Collect data every 100ms
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setConversationState('listening');
      
      // Start checking audio levels
      checkAudioLevel();
      
      // Add a timeout safety net
      setTimeout(() => {
        if (isRecording && mediaRecorderRef.current?.state === 'recording') {
          console.log('Max recording time reached, stopping...');
          stopRecording();
        }
      }, 30000); // 30 seconds max
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast({
        title: "Microphone Error",
        description: "Could not access microphone. Please check permissions.",
        variant: "destructive"
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processVoiceChat = async (audioBlob: Blob) => {
    console.log('Processing voice chat, blob size:', audioBlob.size);
    setIsProcessing(true);
    setConversationState('processing');
    
    try {
      // Create form data for voice chat
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('conversation_id', conversationId);
      formData.append('use_memory', 'true');
      formData.append('use_search', useSearch.toString());
      if (model) {
        formData.append('model', model);
      }
      
      console.log('Sending voice chat request...');

      const response = await fetch('/api/voice/voice-chat', {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('Voice chat failed');
      }

      // Get response headers (decode URL encoding)
      const spokenText = decodeURIComponent(response.headers.get('X-Response-Text') || '');
      const fullResponse = decodeURIComponent(response.headers.get('X-Full-Response') || spokenText);
      const hasCode = response.headers.get('X-Has-Code') === 'true';
      const waitForResponse = response.headers.get('X-Wait-For-Response') === 'true';
      
      console.log('Voice response received:', { spokenText, hasCode, waitForResponse });
      
      // Get audio data
      const audioData = await response.blob();
      const audioUrl = URL.createObjectURL(audioData);

      // Parse code blocks from full response if present
      const codeBlocks = extractCodeBlocks(fullResponse);

      const voiceResponse: VoiceResponse = {
        spokenText,
        fullResponse,
        hasCode,
        codeBlocks,
        audioUrl,
        waitForResponse
      };

      setLastResponse(voiceResponse);
      setConversationState(waitForResponse ? 'waiting' : 'speaking');
      
      // Refresh conversation to show new messages
      refetch();

      // If waiting for response, set up listener
      if (waitForResponse) {
        setTimeout(() => {
          setConversationState('idle');
          // Could auto-start recording here if desired
        }, 2000);
      }
      
    } catch (error) {
      console.error('Error processing voice chat:', error);
      toast({
        title: "Voice Chat Error",
        description: "Failed to process voice conversation. Please try again.",
        variant: "destructive"
      });
      setConversationState('idle');
    } finally {
      setIsProcessing(false);
    }
  };

  const extractCodeBlocks = (text: string): Array<{ language: string; code: string }> => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const blocks: Array<{ language: string; code: string }> = [];
    let match;

    while ((match = codeBlockRegex.exec(text)) !== null) {
      blocks.push({
        language: match[1] || 'plaintext',
        code: match[2].trim()
      });
    }

    return blocks;
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const getStateIcon = () => {
    switch (conversationState) {
      case 'listening':
        return <MicOff className="w-5 h-5" />;
      case 'processing':
        return <Loader2 className="w-5 h-5 animate-spin" />;
      case 'speaking':
        return <Volume2 className="w-5 h-5 animate-pulse" />;
      case 'waiting':
        return <MessageSquare className="w-5 h-5" />;
      default:
        return <Mic className="w-5 h-5" />;
    }
  };

  const getStateText = () => {
    switch (conversationState) {
      case 'listening':
        return "Listening... (I'll stop when you pause)";
      case 'processing':
        return "Processing your request...";
      case 'speaking':
        return "ADAM is speaking";
      case 'waiting':
        return "Your turn to respond";
      default:
        return "Click to start voice conversation";
    }
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <Mic className="w-5 h-5" />
          Voice Conversation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Recording Button */}
        <div className="flex flex-col items-center gap-4">
          <Button
            variant={isRecording ? "destructive" : "default"}
            size="lg"
            onClick={toggleRecording}
            disabled={isProcessing}
            className={cn(
              "relative w-32 h-32 rounded-full",
              isRecording && "animate-pulse"
            )}
          >
            {getStateIcon()}
            
            {isRecording && (
              <span className="absolute -top-2 -right-2 w-4 h-4 bg-red-500 rounded-full animate-pulse" />
            )}
          </Button>
          
          <p className="text-sm text-muted-foreground text-center">
            {getStateText()}
          </p>
        </div>

        {/* Last Response */}
        {lastResponse && (
          <div className="space-y-3 border-t pt-4">
            {/* Spoken Response with Audio Player */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium flex items-center gap-2">
                <Volume2 className="w-4 h-4" />
                What ADAM said:
              </h4>
              <p className="text-sm text-muted-foreground">{lastResponse.spokenText}</p>
              {lastResponse.audioUrl && (
                <AudioPlayer 
                  audioUrl={lastResponse.audioUrl}
                  autoPlay={true}
                  onEnded={() => setConversationState(lastResponse.waitForResponse ? 'waiting' : 'idle')}
                />
              )}
            </div>

            {/* Code Blocks if Present */}
            {lastResponse.hasCode && lastResponse.codeBlocks && lastResponse.codeBlocks.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium flex items-center gap-2">
                  <Code2 className="w-4 h-4" />
                  Code Snippets:
                </h4>
                {lastResponse.codeBlocks.map((block, index) => (
                  <CodeBlock
                    key={index}
                    code={block.code}
                    language={block.language}
                  />
                ))}
              </div>
            )}

            {/* Toggle Full Response */}
            {lastResponse.fullResponse !== lastResponse.spokenText && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowFullResponse(!showFullResponse)}
                className="w-full"
              >
                {showFullResponse ? "Hide" : "Show"} Full Response
              </Button>
            )}

            {/* Full Response */}
            {showFullResponse && (
              <div className="p-3 bg-muted/50 rounded-lg">
                <pre className="whitespace-pre-wrap text-sm">{lastResponse.fullResponse}</pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}