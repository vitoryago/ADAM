/**
 * Example: Adding Voice Conversation to Your Chat Interface
 * 
 * This example shows how to integrate ADAM's voice conversation
 * into an existing chat application.
 */

import { useState } from "react";
import { VoiceConversation } from "@/components/chat/voice-conversation";
import { Button } from "@/components/ui/button";
import { Mic, MicOff } from "lucide-react";

// Example 1: Simple Voice Toggle Button
export function ChatWithVoiceToggle({ conversationId }: { conversationId: string }) {
  const [voiceMode, setVoiceMode] = useState(false);

  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Your existing chat messages */}
      </div>

      {/* Toggle Voice Mode */}
      <div className="border-t p-4">
        <Button
          variant={voiceMode ? "default" : "outline"}
          onClick={() => setVoiceMode(!voiceMode)}
          className="mb-4"
        >
          {voiceMode ? <MicOff className="w-4 h-4 mr-2" /> : <Mic className="w-4 h-4 mr-2" />}
          {voiceMode ? "Exit Voice Mode" : "Enter Voice Mode"}
        </Button>

        {voiceMode ? (
          <VoiceConversation 
            conversationId={conversationId}
            model="grok-3-mini-fast"
            useSearch={true}
          />
        ) : (
          // Your regular text input
          <textarea 
            className="w-full p-2 border rounded"
            placeholder="Type your message..."
          />
        )}
      </div>
    </div>
  );
}

// Example 2: Inline Voice Button
export function ChatWithInlineVoice({ conversationId }: { conversationId: string }) {
  const [isRecording, setIsRecording] = useState(false);

  const handleVoiceInput = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('conversation_id', conversationId);

    const response = await fetch('/api/voice/voice-chat', {
      method: 'POST',
      body: formData,
    });

    // Handle response...
  };

  return (
    <div className="flex items-center gap-2 p-4 border-t">
      <textarea 
        className="flex-1 p-2 border rounded"
        placeholder="Type a message..."
      />
      
      <Button
        variant={isRecording ? "destructive" : "outline"}
        size="icon"
        onClick={() => setIsRecording(!isRecording)}
      >
        {isRecording ? <MicOff /> : <Mic />}
      </Button>
      
      <Button>Send</Button>
    </div>
  );
}

// Example 3: Full Voice Integration with Settings
export function AdvancedVoiceChat({ conversationId }: { conversationId: string }) {
  const [voiceSettings, setVoiceSettings] = useState({
    enabled: true,
    autoListen: false,
    model: "grok-3-mini-fast",
    useSearch: false,
    voiceId: "default"
  });

  return (
    <div className="grid grid-cols-3 gap-4 h-full">
      {/* Chat Area */}
      <div className="col-span-2">
        {/* Chat messages */}
      </div>

      {/* Voice Panel */}
      <div className="border-l p-4">
        <h3 className="font-semibold mb-4">Voice Assistant</h3>
        
        {voiceSettings.enabled ? (
          <VoiceConversation
            conversationId={conversationId}
            model={voiceSettings.model}
            useSearch={voiceSettings.useSearch}
            className="h-full"
          />
        ) : (
          <Button 
            onClick={() => setVoiceSettings({...voiceSettings, enabled: true})}
            className="w-full"
          >
            Enable Voice Mode
          </Button>
        )}

        {/* Voice Settings */}
        <div className="mt-4 space-y-2 text-sm">
          <label className="flex items-center gap-2">
            <input 
              type="checkbox"
              checked={voiceSettings.autoListen}
              onChange={(e) => setVoiceSettings({
                ...voiceSettings, 
                autoListen: e.target.checked
              })}
            />
            Auto-listen after response
          </label>
          
          <label className="flex items-center gap-2">
            <input 
              type="checkbox"
              checked={voiceSettings.useSearch}
              onChange={(e) => setVoiceSettings({
                ...voiceSettings, 
                useSearch: e.target.checked
              })}
            />
            Enable web search
          </label>
        </div>
      </div>
    </div>
  );
}

// Example 4: Voice-First Interface
export function VoiceFirstChat({ projectId }: { projectId: string }) {
  const [conversation, setConversation] = useState<string | null>(null);

  const startNewVoiceChat = async () => {
    // Create new conversation
    const response = await fetch(`/api/projects/${projectId}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: `Voice Chat ${new Date().toLocaleString()}`,
        metadata: { type: 'voice' }
      })
    });
    
    const data = await response.json();
    setConversation(data.id);
  };

  if (!conversation) {
    return (
      <div className="flex items-center justify-center h-full">
        <Button size="lg" onClick={startNewVoiceChat}>
          <Mic className="w-6 h-6 mr-2" />
          Start Voice Conversation
        </Button>
      </div>
    );
  }

  return (
    <VoiceConversation
      conversationId={conversation}
      model="grok-4"
      useSearch={true}
      className="max-w-2xl mx-auto"
    />
  );
}

// Example 5: Custom Voice Response Handler
export function CustomVoiceHandler({ conversationId }: { conversationId: string }) {
  const handleVoiceResponse = (response: any) => {
    // Custom handling of voice responses
    if (response.hasCode) {
      // Show code in a special panel
      console.log('Code blocks detected:', response.codeBlocks);
    }
    
    if (response.waitForResponse) {
      // Auto-start listening again
      setTimeout(() => {
        // Start recording...
      }, 1000);
    }
  };

  return (
    <VoiceConversation
      conversationId={conversationId}
      onResponse={handleVoiceResponse}
    />
  );
}