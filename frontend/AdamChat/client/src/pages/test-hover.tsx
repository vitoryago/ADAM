import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Edit2, Trash2 } from "lucide-react";

export function TestHoverPage() {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  const conversations = [
    { id: "1", title: "Test Conversation 1", updatedAt: new Date() },
    { id: "2", title: "Test Conversation 2", updatedAt: new Date() },
    { id: "3", title: "Test Conversation 3", updatedAt: new Date() },
  ];

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Test Hover States</h1>
      
      <div className="space-y-4">
        <div className="p-4 bg-muted rounded">
          <h2 className="font-semibold mb-2">Test 1: Basic CSS Hover</h2>
          <div className="group p-3 bg-background rounded hover:bg-muted cursor-pointer">
            <div className="flex justify-between items-center">
              <span>Hover me</span>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                <Button size="icon" variant="ghost" className="h-6 w-6">
                  <Edit2 className="w-3 h-3" />
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 bg-muted rounded">
          <h2 className="font-semibold mb-2">Test 2: JavaScript Hover</h2>
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className="p-3 bg-background rounded hover:bg-muted cursor-pointer mb-2"
              onMouseEnter={() => setHoveredItem(conv.id)}
              onMouseLeave={() => setHoveredItem(null)}
            >
              <div className="flex justify-between items-center">
                <span>{conv.title}</span>
                <div 
                  style={{
                    opacity: hoveredItem === conv.id ? 1 : 0,
                    transition: 'opacity 0.2s ease-in-out'
                  }}
                  className="flex gap-1"
                >
                  <Button size="icon" variant="ghost" className="h-6 w-6">
                    <Edit2 className="w-3 h-3" />
                  </Button>
                  <Button size="icon" variant="ghost" className="h-6 w-6">
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 bg-muted rounded">
          <h2 className="font-semibold mb-2">Test 3: Always Visible (Debug)</h2>
          <div className="p-3 bg-background rounded">
            <div className="flex justify-between items-center">
              <span>Always visible buttons</span>
              <div className="flex gap-1">
                <Button size="icon" variant="ghost" className="h-6 w-6">
                  <Edit2 className="w-3 h-3" />
                </Button>
                <Button size="icon" variant="ghost" className="h-6 w-6 text-destructive">
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}