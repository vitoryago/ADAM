export function MarkdownTest() {
  return (
    <div className="p-8">
      <h1 className="text-2xl mb-4">Markdown Style Test</h1>
      
      <div className="space-y-4">
        <div className="border p-4 rounded">
          <h2 className="text-lg mb-2">Direct HTML Test</h2>
          <p>
            This is <strong className="font-bold">bold text</strong> and 
            this is <code className="inline-block rounded bg-muted/80 px-[0.3rem] py-[0.1rem] font-mono text-sm border border-border/50">inline code</code>.
          </p>
        </div>
        
        <div className="border p-4 rounded">
          <h2 className="text-lg mb-2">Font Weight Test</h2>
          <p className="font-normal">Normal (400)</p>
          <p className="font-medium">Medium (500)</p>
          <p className="font-semibold">Semibold (600)</p>
          <p className="font-bold">Bold (700)</p>
          <p className="font-extrabold">Extra Bold (800)</p>
        </div>
        
        <div className="border p-4 rounded">
          <h2 className="text-lg mb-2">Background Test</h2>
          <span className="bg-muted p-2">bg-muted</span>
          <span className="bg-muted/80 p-2 ml-2">bg-muted/80</span>
          <span className="bg-gray-200 p-2 ml-2">bg-gray-200</span>
        </div>
      </div>
    </div>
  );
}
EOF < /dev/null