import { useState } from "react";
import { VoiceConversation } from "@/components/chat/voice-conversation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Mic, Zap, Search, Brain } from "lucide-react";

// This would come from your conversation API
const DEMO_CONVERSATION_ID = "7298ab6d-b1d3-4625-998d-b959b6975ef5";

const AVAILABLE_MODELS = [
  { value: "grok-4", label: "Grok-4 (Most Capable)" },
  { value: "grok-3-mini-fast", label: "Grok-3 Mini Fast" },
  { value: "gpt-4", label: "GPT-4" },
  { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo" },
];

export default function VoiceDemoPage() {
  const [selectedModel, setSelectedModel] = useState("grok-3-mini-fast");
  const [useSearch, setUseSearch] = useState(false);

  return (
    <div className="container mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold flex items-center justify-center gap-3">
          <Mic className="w-10 h-10 text-primary" />
          ADAM Voice Conversation
        </h1>
        <p className="text-lg text-muted-foreground">
          Natural voice interactions with intelligent response filtering
        </p>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Smart Responses
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CardDescription>
              ADAM knows what to speak aloud versus what to display visually. 
              Code snippets are shown, not read.
            </CardDescription>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Natural Flow
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CardDescription>
              Automatic silence detection stops recording after you pause. 
              Natural conversation timing.
            </CardDescription>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Search className="w-5 h-5" />
              Live Search
            </CardTitle>
          </CardHeader>
          <CardContent>
            <CardDescription>
              Enable web search for real-time information. 
              ADAM can look up current data while conversing.
            </CardDescription>
          </CardContent>
        </Card>
      </div>

      {/* Settings */}
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>Voice Settings</CardTitle>
          <CardDescription>
            Configure how ADAM responds to your voice
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label>AI Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AVAILABLE_MODELS.map((model) => (
                  <SelectItem key={model.value} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Search Toggle */}
          <div className="flex items-center justify-between">
            <Label htmlFor="search-toggle" className="flex items-center gap-2">
              <Search className="w-4 h-4" />
              Enable Live Web Search
            </Label>
            <Switch
              id="search-toggle"
              checked={useSearch}
              onCheckedChange={setUseSearch}
            />
          </div>
        </CardContent>
      </Card>

      {/* Voice Conversation Component */}
      <div className="max-w-2xl mx-auto">
        <VoiceConversation
          conversationId={DEMO_CONVERSATION_ID}
          model={selectedModel}
          useSearch={useSearch}
        />
      </div>

      {/* Instructions */}
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle>How to Use Voice Conversation</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            1. <strong>Click the microphone button</strong> to start recording
          </p>
          <p>
            2. <strong>Speak naturally</strong> - ADAM will stop listening when you pause for 1 second
          </p>
          <p>
            3. <strong>Wait for ADAM's response</strong> - both spoken and any visual content
          </p>
          <p>
            4. <strong>Continue the conversation</strong> - ADAM remembers context
          </p>
          
          <div className="bg-muted/50 p-4 rounded-lg mt-4">
            <p className="font-medium mb-2">Example Voice Commands:</p>
            <ul className="space-y-1 text-muted-foreground">
              <li>• "Help me debug this TypeError in my React component"</li>
              <li>• "What's the best way to implement authentication in Next.js?"</li>
              <li>• "Search for the latest Python async best practices"</li>
              <li>• "Explain how to optimize database queries"</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}