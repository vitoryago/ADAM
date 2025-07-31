import { MessageContent } from "@/lib/message-parser";

export default function TestMarkdown() {
  const testCases = [
    "This is **bold text** test",
    "This is `inline code` test",
    "Multiple **bold** and `code` in same line",
    "**Bold at start**",
    "`code at start`",
    "End with **bold**",
    "End with `code`"
  ];

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Markdown Test Page</h1>
      
      <div className="space-y-6">
        {testCases.map((test, i) => (
          <div key={i} className="border rounded-lg p-4">
            <div className="text-sm text-muted-foreground mb-2">Input:</div>
            <pre className="bg-muted p-2 rounded mb-4">{test}</pre>
            
            <div className="text-sm text-muted-foreground mb-2">Rendered:</div>
            <div className="prose-chat">
              <MessageContent content={test} />
            </div>
            
            <div className="text-sm text-muted-foreground mb-2 mt-4">Direct HTML comparison:</div>
            <div className="flex gap-4">
              <strong className="font-bold">font-bold</strong>
              <strong className="font-semibold">font-semibold</strong>
              <code className="inline-block rounded bg-muted/80 px-[0.3rem] py-[0.1rem] font-mono text-sm border border-border/50">inline code</code>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-8 border rounded-lg p-4">
        <h2 className="text-lg font-semibold mb-4">CSS Classes Test</h2>
        <div className="space-y-2">
          <div>Normal text</div>
          <div className="font-medium">font-medium (500)</div>
          <div className="font-semibold">font-semibold (600)</div>
          <div className="font-bold">font-bold (700)</div>
          <div className="font-extrabold">font-extrabold (800)</div>
        </div>
      </div>
    </div>
  );
}