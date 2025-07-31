import { MessageContent } from './lib/message-parser';

export function TestMarkdown() {
  const testContent = "This is **bold text** and this is `inline code` test";
  
  return (
    <div className="p-4">
      <h2>Test Markdown Rendering</h2>
      <div className="border p-4">
        <h3>Raw content:</h3>
        <pre>{testContent}</pre>
      </div>
      <div className="border p-4 mt-4">
        <h3>Rendered with MessageContent:</h3>
        <MessageContent content={testContent} />
      </div>
      <div className="border p-4 mt-4">
        <h3>Direct rendering test:</h3>
        <p>This is <strong className="font-semibold">bold text</strong> and this is <code className="bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm">inline code</code> test</p>
      </div>
    </div>
  );
}
EOF < /dev/null