import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, MicOff, Loader2, Volume2, Wifi } from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { AudioRecorder } from "@/utils/audio-recorder";

interface StreamingVoiceConversationProps {
  conversationId: string;
  className?: string;
}

export function StreamingVoiceConversation({ 
  conversationId, 
  className 
}: StreamingVoiceConversationProps) {
  const { toast } = useToast();
  
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTranscription, setCurrentTranscription] = useState("");
  const [streamingTranscription, setStreamingTranscription] = useState("");
  const [currentResponse, setCurrentResponse] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);
  const audioQueueRef = useRef<{ audio: HTMLAudioElement; sequence: number }[]>([]);
  const isPlayingRef = useRef(false);
  const playQueueRef = useRef<{ data: string; sequence: number }[]>([]);
  const recordingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const silenceDetectorRef = useRef<{ interval?: NodeJS.Timeout; context?: AudioContext } | null>(null);
  
  // Connect to WebSocket on mount
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      disconnectWebSocket();
      // Clean up any playing audio
      playQueueRef.current = [];
      isPlayingRef.current = false;
    };
  }, [conversationId]);
  
  const connectWebSocket = () => {
    const wsUrl = `ws://localhost:8000/ws/voice-stream/${conversationId}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
    };
    
    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      setIsProcessing(false);
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast({
        title: "Connection Error",
        description: "Failed to connect to voice server",
        variant: "destructive"
      });
    };
    
    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      await handleWebSocketMessage(data);
    };
    
    wsRef.current = ws;
  };
  
  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };
  
  const handleWebSocketMessage = async (data: any) => {
    switch (data.type) {
      case "transcription":
        if (data.final) {
          setCurrentTranscription(data.text);
          setStreamingTranscription("");
          setCurrentResponse(""); // Clear previous response
        } else {
          // Streaming transcription update
          setStreamingTranscription(data.text);
        }
        break;
        
      case "text_chunk":
        setCurrentResponse(prev => prev + data.content);
        break;
        
      case "audio_chunk":
        // Play audio chunk with sequence
        await playAudioChunk(data.data, data.sequence);
        if (data.text && data.chunk === 1) {
          console.log("Speaking:", data.text);
        }
        break;
        
      case "completion":
        setIsProcessing(false);
        console.log("Response complete");
        break;
        
      case "error":
        console.error("Server error:", data.message);
        toast({
          title: "Error",
          description: data.message,
          variant: "destructive"
        });
        setIsProcessing(false);
        break;
    }
  };
  
  const playAudioChunk = async (base64Audio: string, sequence?: number) => {
    // Add to play queue
    playQueueRef.current.push({ data: base64Audio, sequence: sequence || 0 });
    
    // Start playing if not already playing
    if (!isPlayingRef.current) {
      processAudioQueue();
    }
  };
  
  const processAudioQueue = async () => {
    if (playQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      return;
    }
    
    isPlayingRef.current = true;
    setIsPlaying(true);
    
    // Get next audio chunk
    const { data, sequence } = playQueueRef.current.shift()!;
    
    try {
      // Create a blob from the base64 data
      const binaryString = atob(data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Create blob with MP3 mime type
      const blob = new Blob([bytes], { type: 'audio/mp3' });
      const url = URL.createObjectURL(blob);
      
      // Create audio element
      const audio = new Audio(url);
      audio.volume = 0.9;
      audio.playbackRate = 0.9; // Slow down playback slightly
      
      // Wait for audio to be ready
      await new Promise((resolve, reject) => {
        audio.oncanplaythrough = resolve;
        audio.onerror = reject;
        audio.load();
      });
      
      // Play the audio
      await audio.play();
      
      // Wait for audio to finish
      await new Promise((resolve) => {
        audio.onended = () => {
          URL.revokeObjectURL(url);
          resolve(undefined);
        };
      });
      
      // Small gap between chunks for natural speech
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Process next chunk
      processAudioQueue();
      
    } catch (error) {
      console.error("Error playing audio chunk:", error);
      // Continue with next chunk even if there's an error
      processAudioQueue();
    }
  };
  
  const startRecording = async () => {
    try {
      // Create new audio recorder
      const recorder = new AudioRecorder();
      audioRecorderRef.current = recorder;
      
      await recorder.startRecording();
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      // Set up silence detection with Web Audio API
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      let silenceStart: number | null = null;
      let hasSpoken = false;
      let checkInterval: NodeJS.Timeout;
      const recordingStartTime = Date.now();
      
      const checkSilence = () => {
        if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
          clearInterval(checkInterval);
          audioContext.close();
          return;
        }
        
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const average = sum / bufferLength;
        
        // More sensitive threshold
        const threshold = 15;
        
        if (average > threshold) {
          hasSpoken = true;
          silenceStart = null;
          console.log(`Speaking detected: volume ${average.toFixed(2)}`);
        } else if (hasSpoken) {
          if (!silenceStart) {
            silenceStart = Date.now();
            console.log(`Silence started at ${new Date().toLocaleTimeString()}`);
          } else {
            const silenceDuration = Date.now() - silenceStart;
            const totalRecordingTime = Date.now() - recordingStartTime;
            console.log(`Silence duration: ${silenceDuration}ms, total recording: ${totalRecordingTime}ms`);
            
            // Ensure minimum recording time of 2 seconds for valid audio
            if (silenceDuration >= 1000 && totalRecordingTime >= 2000) {
              console.log(`Stopping recording after ${silenceDuration}ms of silence`);
              clearInterval(checkInterval);
              stopRecording();
              audioContext.close();
            }
          }
        }
      };
      
      // Check every 100ms
      checkInterval = setInterval(checkSilence, 100);
      
      mediaRecorder.start(100); // Collect chunks every 100ms for better audio quality
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      
      // Set max recording time
      recordingTimeoutRef.current = setTimeout(() => {
        console.log("Max recording time reached");
        stopRecording();
      }, 30000);
      
      // Ensure minimum recording time for valid audio
      setTimeout(() => {
        console.log("Minimum recording time reached, silence detection active");
      }, 500); // At least 500ms of audio
      
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
  
  const sendAudioToServer = async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not connected");
      return;
    }
    
    if (audioChunksRef.current.length === 0) return;
    
    // Combine chunks into single blob with proper mime type
    const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm';
    const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
    audioChunksRef.current = []; // Clear chunks
    
    console.log(`Sending audio blob: size=${audioBlob.size}, type=${audioBlob.type}`);
    
    // Convert to base64
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64Audio = reader.result?.toString().split(',')[1];
      
      if (base64Audio) {
        console.log(`Base64 audio length: ${base64Audio.length} chars`);
        console.log(`First 100 chars of base64: ${base64Audio.substring(0, 100)}`);
        
        setIsProcessing(true);
        // Send final audio with format hint
        wsRef.current!.send(JSON.stringify({
          type: "audio",
          data: base64Audio,
          format: mimeType.includes('webm') ? 'webm' : 'ogg',
          mimeType: mimeType,
          final: true
        }));
      }
    };
    reader.readAsDataURL(audioBlob);
  };
  
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };
  
  const getStatusText = () => {
    if (!isConnected) return "Connecting...";
    if (isRecording) return "Listening... (stop talking for 1 second to send)";
    if (isProcessing) return "Processing...";
    if (isPlaying) return "ADAM is speaking...";
    return "Click to start talking";
  };
  
  const getStatusIcon = () => {
    if (!isConnected) return <Wifi className="w-5 h-5 animate-pulse" />;
    if (isRecording) return <MicOff className="w-5 h-5" />;
    if (isProcessing) return <Loader2 className="w-5 h-5 animate-spin" />;
    if (isPlaying) return <Volume2 className="w-5 h-5 animate-pulse" />;
    return <Mic className="w-5 h-5" />;
  };
  
  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <Mic className="w-5 h-5" />
          Streaming Voice Mode (Real-time)
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Status indicator */}
        {!isConnected && (
          <div className="text-sm text-muted-foreground text-center">
            Connecting to voice server...
          </div>
        )}
        
        {/* Recording Button */}
        <div className="flex flex-col items-center gap-4">
          <Button
            variant={isRecording ? "destructive" : "default"}
            size="lg"
            onClick={toggleRecording}
            disabled={!isConnected || isProcessing}
            className={cn(
              "relative w-32 h-32 rounded-full",
              isRecording && "animate-pulse"
            )}
          >
            {getStatusIcon()}
            
            {isRecording && (
              <span className="absolute -top-2 -right-2 w-4 h-4 bg-red-500 rounded-full animate-pulse" />
            )}
          </Button>
          
          <p className="text-sm text-muted-foreground text-center">
            {getStatusText()}
          </p>
        </div>
        
        {/* Transcription */}
        {(streamingTranscription || currentTranscription) && (
          <div className="space-y-2 border-t pt-4">
            <h4 className="text-sm font-medium">You said:</h4>
            <p className="text-sm text-muted-foreground">
              {streamingTranscription || currentTranscription}
              {streamingTranscription && <span className="animate-pulse">...</span>}
            </p>
          </div>
        )}
        
        {/* Response */}
        {currentResponse && (
          <div className="space-y-2 border-t pt-4">
            <h4 className="text-sm font-medium flex items-center gap-2">
              {isPlaying && <Volume2 className="w-4 h-4 animate-pulse" />}
              ADAM:
            </h4>
            <div className="text-sm whitespace-pre-wrap">{currentResponse}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}