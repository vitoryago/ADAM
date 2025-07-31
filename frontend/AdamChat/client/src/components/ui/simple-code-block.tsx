import { useState } from 'react';
import { Copy, Check, Code2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  code: string;
  language?: string;
  inline?: boolean;
  className?: string;
}

export function CodeBlock({ code, language, inline = false, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: 'Code copied',
        description: 'Code snippet copied to clipboard.',
      });
    } catch (error) {
      toast({
        title: 'Copy failed',
        description: 'Failed to copy code to clipboard.',
        variant: 'destructive',
      });
    }
  };

  if (inline) {
    return (
      <code className={cn(
        'inline-block rounded bg-muted/80 px-[0.3rem] py-[0.1rem] font-mono text-sm',
        'border border-border/50',
        'text-foreground',
        className
      )}>
        {code}
      </code>
    );
  }

  return (
    <div className={cn('relative group my-4 w-full', className)}>
      <div className="flex items-center justify-between bg-muted/50 border border-border rounded-t-lg px-4 py-2">
        <div className="flex items-center gap-2">
          <Code2 className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-muted-foreground capitalize">
            {language || 'code'}
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="opacity-0 group-hover:opacity-100 h-7 px-2 transition-opacity"
          onClick={handleCopy}
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
          <span className="ml-1 text-xs">
            {copied ? 'Copied!' : 'Copy'}
          </span>
        </Button>
      </div>
      
      <div className="border border-t-0 border-border rounded-b-lg overflow-auto">
        <pre className="p-4 m-0 text-sm bg-muted/30">
          <code className="font-mono whitespace-pre">{code}</code>
        </pre>
      </div>
    </div>
  );
}