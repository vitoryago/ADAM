import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';
import { Copy, Check, Code2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { useTheme } from 'next-themes';

interface CodeBlockProps {
  code: string;
  language?: string;
  inline?: boolean;
  className?: string;
}

// Language detection patterns
const LANGUAGE_PATTERNS = {
  javascript: /\b(function|const|let|var|=>|import|export|class|extends)\b/,
  typescript: /\b(interface|type|enum|implements|private|public|protected)\b|:\s*\w+/,
  python: /\b(def|import|from|class|if __name__|print|range)\b|^\s*#/,
  jsx: /<[A-Z]\w*|<\/[A-Z]\w*>|className=|onClick=/,
  tsx: /(<[A-Z]\w*|interface|type).*:/,
  css: /\{[^}]*:[^}]*\}|@media|\.[\w-]+\s*\{/,
  scss: /\$[\w-]+:|@import|@mixin|@include/,
  html: /<\/?[a-zA-Z][\w-]*>|<!DOCTYPE/,
  json: /^\s*[\{\[][\s\S]*[\}\]]\s*$/,
  sql: /\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|TABLE)\b/i,
  bash: /^\s*[\w-]+\$|^\s*sudo|^\s*cd\s+|^\s*ls\s*/,
  yaml: /^\s*[\w-]+:\s*|^\s*-\s+/,
  xml: /<\?xml|<\/?\w+[^>]*>/,
  markdown: /^#+\s+|^\*\*|^\*|^\[.*\]\(.*\)/,
  php: /<\?php|^\s*\$\w+/,
  java: /\b(public|private|protected|class|interface|extends|implements)\b.*\{/,
  csharp: /\b(using|namespace|public|private|class|interface)\b/,
  go: /\b(package|import|func|var|type|struct)\b/,
  rust: /\b(fn|let|mut|impl|struct|enum|use)\b/,
  swift: /\b(func|var|let|class|struct|enum|import)\b/,
  kotlin: /\b(fun|val|var|class|interface|object)\b/,
  ruby: /\b(def|class|module|require|puts)\b/,
  perl: /^\s*#!.*perl|^\s*use\s+\w+/,
  lua: /\b(function|local|end|require)\b/,
  r: /\b(library|data\.frame|function|<-)\b/,
  matlab: /\b(function|end|plot|figure)\b/,
};

const detectLanguage = (code: string): string => {
  const trimmedCode = code.trim();
  
  // Try to detect language by patterns
  for (const [lang, pattern] of Object.entries(LANGUAGE_PATTERNS)) {
    if (pattern.test(trimmedCode)) {
      return lang;
    }
  }
  
  // Fallback detection by common extensions or keywords
  if (trimmedCode.includes('import React') || trimmedCode.includes('useState')) {
    return 'jsx';
  }
  
  if (trimmedCode.startsWith('{') && trimmedCode.endsWith('}')) {
    try {
      JSON.parse(trimmedCode);
      return 'json';
    } catch {}
  }
  
  return 'text';
};

export function CodeBlock({ code, language, inline = false, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();
  const { theme } = useTheme();
  
  
  const detectedLanguage = language || detectLanguage(code);
  const isValidLanguage = detectedLanguage !== 'text';
  
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
        'relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm',
        'before:content-[""] before:absolute before:inset-0 before:rounded before:bg-gradient-to-r',
        'before:from-blue-500/10 before:to-purple-500/10 before:opacity-50',
        isValidLanguage && 'text-blue-600 dark:text-blue-400',
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
            {isValidLanguage ? detectedLanguage : 'plain text'}
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
      
      <div className="border border-t-0 border-border rounded-b-lg overflow-hidden w-full">
        {isValidLanguage ? (
          <SyntaxHighlighter
            language={detectedLanguage}
            style={theme === 'dark' ? oneDark : oneLight}
            customStyle={{
              margin: 0,
              borderRadius: 0,
              background: 'transparent',
              fontSize: '14px',
              lineHeight: '1.5',
              whiteSpace: 'pre',
              wordBreak: 'break-word',
              overflowWrap: 'break-word',
            }}
            showLineNumbers={code.split('\n').length > 5}
            lineNumberStyle={{
              minWidth: '3em',
              paddingRight: '1em',
              color: 'var(--muted-foreground)',
              borderRight: '1px solid var(--border)',
              marginRight: '1em',
            }}
            wrapLines={true}
            wrapLongLines={true}
            PreTag="div"
          >
            {code}
          </SyntaxHighlighter>
        ) : (
          <pre className="p-4 text-sm overflow-x-auto bg-muted/20 whitespace-pre">
            <code className="whitespace-pre">{code}</code>
          </pre>
        )}
      </div>
    </div>
  );
}

// Enhanced inline code component
interface InlineCodeProps {
  children: string;
  className?: string;
}

export function InlineCode({ children, className }: InlineCodeProps) {
  return (
    <CodeBlock 
      code={children} 
      inline={true} 
      className={className}
    />
  );
}