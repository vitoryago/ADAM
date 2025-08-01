import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Play, Pause, Volume2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

interface AudioPlayerProps {
  audioUrl: string;
  autoPlay?: boolean;
  onEnded?: () => void;
  className?: string;
}

export function AudioPlayer({ 
  audioUrl, 
  autoPlay = false, 
  onEnded,
  className 
}: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const audio = new Audio(audioUrl);
    audio.volume = volume;
    
    audio.onloadedmetadata = () => {
      setDuration(audio.duration);
    };
    
    audio.onended = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      if (onEnded) {
        onEnded();
      }
    };
    
    audio.onplay = () => {
      setIsPlaying(true);
    };
    
    audio.onpause = () => {
      setIsPlaying(false);
    };
    
    audio.ontimeupdate = () => {
      setCurrentTime(audio.currentTime);
    };

    audioRef.current = audio;
    
    if (autoPlay) {
      audio.play().catch(console.error);
    }

    return () => {
      audio.pause();
      audio.src = '';
    };
  }, [audioUrl, autoPlay, onEnded, volume]);

  const togglePlayPause = () => {
    if (!audioRef.current) return;

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

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className={cn("flex items-center gap-3 p-3 bg-muted/50 rounded-lg", className)}>
      <Button
        variant="ghost"
        size="icon"
        onClick={togglePlayPause}
        className="shrink-0"
      >
        {isPlaying ? (
          <Pause className="w-4 h-4" />
        ) : (
          <Play className="w-4 h-4" />
        )}
      </Button>
      
      <div className="flex-1 space-y-1">
        <div className="relative h-1 bg-muted rounded-full overflow-hidden">
          <div 
            className="absolute h-full bg-primary transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
      
      <div className="flex items-center gap-2">
        <Volume2 className="w-4 h-4 text-muted-foreground" />
        <Slider
          value={[volume]}
          onValueChange={handleVolumeChange}
          max={1}
          step={0.1}
          className="w-20"
        />
      </div>
    </div>
  );
}