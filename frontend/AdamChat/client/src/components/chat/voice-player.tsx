import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX, Loader2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

interface VoicePlayerProps {
  text: string;
  voiceId?: string;
  autoPlay?: boolean;
  className?: string;
}

export function VoicePlayer({ text, voiceId, autoPlay = false, className }: VoicePlayerProps) {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  useEffect(() => {
    // Cleanup audio URL on unmount
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  useEffect(() => {
    if (autoPlay && text) {
      synthesizeAndPlay();
    }
  }, [text, autoPlay]);

  const synthesizeAndPlay = async () => {
    if (!text) return;
    
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/voice/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          voice_id: voiceId,
          stream: false
        }),
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('Synthesis failed');
      }

      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

      // Create and play audio
      const audio = new Audio(url);
      audio.volume = volume;
      
      audio.onended = () => {
        setIsPlaying(false);
      };
      
      audio.onplay = () => {
        setIsPlaying(true);
      };
      
      audio.onpause = () => {
        setIsPlaying(false);
      };

      audioRef.current = audio;
      await audio.play();
      
    } catch (error) {
      console.error('Error synthesizing speech:', error);
      toast({
        title: "Voice Error",
        description: "Failed to generate speech. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const togglePlayPause = () => {
    if (!audioRef.current) {
      synthesizeAndPlay();
      return;
    }

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0];
    setVolume(newVolume);
    
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
  };

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <Button
        variant="ghost"
        size="icon"
        onClick={togglePlayPause}
        disabled={isLoading || !text}
      >
        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : isPlaying ? (
          <VolumeX className="w-4 h-4" />
        ) : (
          <Volume2 className="w-4 h-4" />
        )}
      </Button>
      
      <Slider
        value={[volume]}
        onValueChange={handleVolumeChange}
        max={1}
        step={0.1}
        className="w-24"
        disabled={isLoading}
      />
    </div>
  );
}