import { MessageContent } from "@/lib/message-parser";

export function TestMessagePage() {
  const testContent = `This is a test message with **bold text** and \`inline code\`.

Here's a code block:
\`\`\`javascript
function hello() {
  console.log("Hello, World!");
}
\`\`\`

And here's a list:
- Item 1
- Item 2 with **bold**
- Item 3 with \`code\`

[This is a link](https://example.com)`;

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Message Parser Test</h1>
      
      <div className="space-y-6">
        <div className="border rounded p-4">
          <h2 className="font-semibold mb-2">Raw Content:</h2>
          <pre className="bg-muted p-3 rounded text-sm overflow-x-auto">
            {testContent}
          </pre>
        </div>

        <div className="border rounded p-4">
          <h2 className="font-semibold mb-2">Rendered Content:</h2>
          <div className="bg-muted p-4 rounded">
            <MessageContent content={testContent} />
          </div>
        </div>

        <div className="border rounded p-4">
          <h2 className="font-semibold mb-2">Simple Tests:</h2>
          <div className="space-y-2">
            <div className="p-2 bg-muted rounded">
              <MessageContent content="This has **bold** text" />
            </div>
            <div className="p-2 bg-muted rounded">
              <MessageContent content="This has `inline code` here" />
            </div>
            <div className="p-2 bg-muted rounded">
              <MessageContent content="**Bold** and `code` together" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TestMessagePage;