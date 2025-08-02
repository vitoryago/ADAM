import { useState, useRef, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, MicOff, Loader2, Volume2, Wifi } from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

interface StreamingVoiceConversationProps {
  conversationId: string;
  className?: string;
  onConversationComplete?: () => void;
}

export function StreamingVoiceConversation({ 
  conversationId, 
  className,
  onConversationComplete 
}: StreamingVoiceConversationProps) {
  const { toast } = useToast();
  
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTranscription, setCurrentTranscription] = useState("");
  const [streamingTranscription, setStreamingTranscription] = useState("");
  const [currentResponse, setCurrentResponse] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [modelInfo, setModelInfo] = useState<{ model?: string; tts?: string } | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playQueueRef = useRef<{ data: string; sequence: number }[]>([]);
  const isPlayingRef = useRef(false);
  
  // Connect to WebSocket on mount
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      disconnectWebSocket();
      // Safely close audio context
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {
          // Ignore errors when closing
        });
      }
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
        setCurrentTranscription(data.text);
        setStreamingTranscription("");
        setCurrentResponse(""); // Clear previous response
        setModelInfo(null); // Clear previous model info
        break;
        
      case "text_chunk":
        setCurrentResponse(prev => prev + data.content);
        break;
        
      case "audio_chunk":
        // Add to play queue
        playQueueRef.current.push({ data: data.data, sequence: data.sequence || 0 });
        if (!isPlayingRef.current) {
          processAudioQueue();
        }
        break;
        
      case "completion":
        setIsProcessing(false);
        console.log("Response complete", data);
        // Set model info
        setModelInfo({
          model: data.model,
          tts: data.tts_provider
        });
        // The conversation history is saved on the backend
        // Notify parent to refresh messages
        if (onConversationComplete) {
          onConversationComplete();
        }
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
  
  const processAudioQueue = async () => {
    if (playQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      setIsPlaying(false);
      return;
    }
    
    isPlayingRef.current = true;
    setIsPlaying(true);
    
    // Process multiple chunks at once for smoother playback
    const chunksToProcess = [];
    const targetSequence = playQueueRef.current[0].sequence;
    
    // Collect all chunks for the same sentence
    while (playQueueRef.current.length > 0 && playQueueRef.current[0].sequence === targetSequence) {
      chunksToProcess.push(playQueueRef.current.shift()!);
    }
    
    try {
      // Combine all chunks for this sentence
      const combinedData = chunksToProcess.map(chunk => {
        const binaryString = atob(chunk.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes;
      });
      
      // Create a single blob from all chunks
      const blob = new Blob(combinedData, { type: 'audio/mp3' });
      const url = URL.createObjectURL(blob);
      
      const audio = new Audio(url);
      audio.volume = 0.9;
      
      await new Promise((resolve, reject) => {
        audio.oncanplaythrough = resolve;
        audio.onerror = reject;
        audio.load();
      });
      
      await audio.play();
      
      await new Promise((resolve) => {
        audio.onended = () => {
          URL.revokeObjectURL(url);
          resolve(undefined);
        };
      });
      
      // Process next sentence immediately
      processAudioQueue();
      
    } catch (error) {
      console.error("Error playing audio:", error);
      processAudioQueue();
    }
  };
  
  const startRecording = async () => {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      
      // Create audio context for processing if not already created
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      
      // Use simple WAV recording approach
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        await processRecordedAudio();
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Start recording
      mediaRecorder.start();
      setIsRecording(true);
      
      // Set up silence detection
      setupSilenceDetection(stream);
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast({
        title: "Microphone Error",
        description: "Could not access microphone. Please check permissions.",
        variant: "destructive"
      });
    }
  };
  
  const setupSilenceDetection = (stream: MediaStream) => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      console.error('AudioContext not available for silence detection');
      return;
    }
    
    const audioContext = audioContextRef.current;
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    let silenceStart: number | null = null;
    let hasSpoken = false;
    const recordingStartTime = Date.now();
    
    const checkSilence = () => {
      if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
        return;
      }
      
      analyser.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      const average = sum / bufferLength;
      
      const threshold = 15;
      
      if (average > threshold) {
        hasSpoken = true;
        silenceStart = null;
        console.log(`Speaking detected: volume ${average.toFixed(2)}`);
      } else if (hasSpoken) {
        if (!silenceStart) {
          silenceStart = Date.now();
          console.log("Silence started");
        } else {
          const silenceDuration = Date.now() - silenceStart;
          const totalRecordingTime = Date.now() - recordingStartTime;
          
          // Stop after 1.5 seconds of silence and minimum 2 seconds recording
          if (silenceDuration >= 1500 && totalRecordingTime >= 2000) {
            console.log(`Stopping after ${silenceDuration}ms of silence`);
            stopRecording();
            return;
          }
        }
      }
      
      // Continue checking
      requestAnimationFrame(checkSilence);
    };
    
    // Start checking
    requestAnimationFrame(checkSilence);
    
    // Max recording time
    setTimeout(() => {
      if (isRecording) {
        console.log("Max recording time reached");
        stopRecording();
      }
    }, 30000);
  };
  
  const processRecordedAudio = async () => {
    if (audioChunksRef.current.length === 0) return;
    
    // Create blob from chunks
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    console.log(`Audio blob size: ${audioBlob.size} bytes`);
    
    // Send WebM directly - let backend handle conversion
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64Audio = reader.result?.toString().split(',')[1];
      if (base64Audio && wsRef.current?.readyState === WebSocket.OPEN) {
        console.log(`Sending audio: ${base64Audio.length} base64 chars`);
        setIsProcessing(true);
        wsRef.current.send(JSON.stringify({
          type: "audio",
          data: base64Audio,
          format: "webm",
          mimeType: "audio/webm",
          final: true
        }));
      }
    };
    reader.readAsDataURL(audioBlob);
  };
  
  const audioBufferToWav = async (audioBuffer: AudioBuffer): Promise<Blob> => {
    const numberOfChannels = 1; // Mono
    const sampleRate = 16000;
    const format = 1; // PCM
    const bitDepth = 16;
    
    // Calculate sizes
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = audioBuffer.length * blockAlign;
    
    // Create buffer
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    
    // Write WAV header
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    // RIFF chunk
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    
    // fmt chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true); // audio format
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    
    // data chunk
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    // Write audio data
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    
    for (let i = 0; i < channelData.length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    
    // Clean up stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
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
    if (isRecording) return "Listening... (pause for 1.5 seconds to send)";
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
        {currentTranscription && (
          <div className="space-y-2 border-t pt-4">
            <h4 className="text-sm font-medium">You said:</h4>
            <p className="text-sm text-muted-foreground">
              {currentTranscription}
            </p>
          </div>
        )}
        
        {/* Response */}
        {currentResponse && (
          <div className="space-y-2 border-t pt-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium flex items-center gap-2">
                {isPlaying && <Volume2 className="w-4 h-4 animate-pulse" />}
                ADAM:
              </h4>
              {modelInfo && (
                <div className="text-xs text-muted-foreground flex items-center gap-2">
                  <span className="px-2 py-1 bg-muted rounded-md">{modelInfo.model || 'Unknown'}</span>
                  <span className="px-2 py-1 bg-muted rounded-md">🔊 {modelInfo.tts || 'ElevenLabs'}</span>
                </div>
              )}
            </div>
            <div className="text-sm whitespace-pre-wrap">{currentResponse}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}