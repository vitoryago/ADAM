import { Button } from "@/components/ui/button";
import { Edit2, Trash2 } from "lucide-react";

export function SidebarTest() {
  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-bold">Sidebar Button Test</h1>
      
      {/* Test 1: Always visible buttons */}
      <div className="border p-4 rounded">
        <h2 className="font-semibold mb-2">Test 1: Always Visible</h2>
        <div className="flex items-center justify-between p-3 bg-muted rounded">
          <span>Conversation Title</span>
          <div className="flex gap-2">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Edit2 className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Test 2: Group hover */}
      <div className="border p-4 rounded">
        <h2 className="font-semibold mb-2">Test 2: Group Hover (hover over the gray area)</h2>
        <div className="group flex items-center justify-between p-3 bg-muted rounded hover:bg-muted/80">
          <span>Hover over me</span>
          <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Edit2 className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Test 3: Direct visibility test */}
      <div className="border p-4 rounded">
        <h2 className="font-semibold mb-2">Test 3: Icons Only</h2>
        <div className="flex gap-4">
          <Edit2 className="w-6 h-6" />
          <Trash2 className="w-6 h-6 text-red-500" />
        </div>
      </div>
    </div>
  );
}