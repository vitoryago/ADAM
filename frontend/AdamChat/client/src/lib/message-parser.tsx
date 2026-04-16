import React from 'react';
import { CodeBlock } from '@/components/ui/code-highlighter';
const InlineCode = ({ children }: { children: React.ReactNode }) => (
  <code className="inline-block rounded bg-muted/80 px-[0.3rem] py-[0.1rem] font-mono text-sm border border-border/50 text-foreground">
    {children}
  </code>
);
import { cn } from '@/lib/utils';

export interface ParsedContent {
  type: 'text' | 'code' | 'inline-code' | 'bold' | 'italic' | 'list' | 'link' | 'linebreak' | 'heading';
  content: string;
  language?: string;
  href?: string;
  level?: number;
}

function renderInlinePart(part: ParsedContent, key: string): React.ReactNode {
  switch (part.type) {
    case 'inline-code':
      return (
        <InlineCode key={key}>
          {part.content}
        </InlineCode>
      );
    case 'bold':
      return (
        <strong key={key} style={{ fontWeight: 700 }}>
          {part.content}
        </strong>
      );
    case 'italic':
      return (
        <em key={key} className="italic">
          {part.content}
        </em>
      );
    case 'link':
      return (
        <a
          key={key}
          href={part.href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-blue-400 hover:underline break-all"
        >
          {part.content}
        </a>
      );
    case 'linebreak':
      return <br key={key} />;
    case 'text':
    default:
      return <React.Fragment key={key}>{part.content}</React.Fragment>;
  }
}

// Enhanced message parser that handles multiple markdown elements
export function parseMessageContent(content: string): ParsedContent[] {
  const parts: ParsedContent[] = [];
  let remaining = content;
  
  while (remaining.length > 0) {
    // Check for code blocks first (highest priority)
    // Match code blocks with optional language identifier
    const codeBlockMatch = remaining.match(/^```(\w+)?\s*\n?([\s\S]*?)```/);
    if (codeBlockMatch) {
      const [fullMatch, language, code] = codeBlockMatch;
      // Don't trim the code to preserve formatting
      const codeContent = code.startsWith('\n') ? code.slice(1) : code;
      
      
      parts.push({
        type: 'code',
        content: codeContent,
        language: language || undefined
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for headings (after code blocks to avoid conflicts)
    const headingMatch = remaining.match(/^(#{1,6})\s+(.+)(?:\n|$)/);
    if (headingMatch) {
      const [fullMatch, hashes, text] = headingMatch;
      parts.push({
        type: 'heading',
        content: text,
        level: hashes.length
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for inline code
    const inlineCodeMatch = remaining.match(/^`([^`\n]+)`/);
    if (inlineCodeMatch) {
      const [fullMatch, code] = inlineCodeMatch;
      parts.push({
        type: 'inline-code',
        content: code
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for bold text (non-greedy, single line)
    const boldMatch = remaining.match(/^\*\*([^*\n]+)\*\*/);
    if (boldMatch) {
      const [fullMatch, text] = boldMatch;
      parts.push({
        type: 'bold',
        content: text
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for italic text (non-greedy, avoid list items)
    const italicMatch = remaining.match(/^\*([^*\n]+)\*/);
    if (italicMatch && !remaining.match(/^[\*\-\+]\s/)) {  // Avoid list items
      const [fullMatch, text] = italicMatch;
      parts.push({
        type: 'italic',
        content: text
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for links, allowing URLs with nested parentheses.
    if (remaining.startsWith('[')) {
      const labelEnd = remaining.indexOf('](');
      if (labelEnd > 0) {
        let depth = 0;
        let hrefEnd = -1;
        for (let i = labelEnd + 2; i < remaining.length; i += 1) {
          const char = remaining[i];
          if (char === '(') depth += 1;
          if (char === ')') {
            if (depth === 0) {
              hrefEnd = i;
              break;
            }
            depth -= 1;
          }
        }

        if (hrefEnd !== -1) {
          const text = remaining.slice(1, labelEnd);
          const href = remaining.slice(labelEnd + 2, hrefEnd);
          const fullMatch = remaining.slice(0, hrefEnd + 1);
          parts.push({
            type: 'link',
            content: text,
            href,
          });
          remaining = remaining.slice(fullMatch.length);
          continue;
        }
      }
    }

    const linkMatch = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch) {
      const [fullMatch, text, href] = linkMatch;
      parts.push({
        type: 'link',
        content: text,
        href: href
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for list items (both unordered and ordered)
    const listMatch = remaining.match(/^(?:[\-\*\+]|\d+\.) (.+)(?:\n|$)/);
    if (listMatch) {
      const [fullMatch, text] = listMatch;
      parts.push({
        type: 'list',
        content: text
      });
      remaining = remaining.slice(fullMatch.length);
      continue;
    }
    
    // Check for line breaks
    if (remaining.startsWith('\n')) {
      parts.push({
        type: 'linebreak',
        content: '\n'
      });
      remaining = remaining.slice(1);
      continue;
    }
    
    // Find the next special character or just take the next character
    const nextSpecialIndex = remaining.search(/[`*\[\-\n#]/);
    if (nextSpecialIndex === -1) {
      // No more special characters, add the rest as text
      parts.push({
        type: 'text',
        content: remaining
      });
      remaining = '';
    } else if (nextSpecialIndex === 0) {
      // Special character at start but no match found, take one character
      parts.push({
        type: 'text',
        content: remaining[0]
      });
      remaining = remaining.slice(1);
    } else {
      // Add text up to the next special character
      parts.push({
        type: 'text',
        content: remaining.slice(0, nextSpecialIndex)
      });
      remaining = remaining.slice(nextSpecialIndex);
    }
  }
  
  return parts;
}

// Component to render parsed content
interface MessageContentProps {
  content: string;
  className?: string;
}

export function MessageContent({ content, className }: MessageContentProps) {
  const parsedParts = parseMessageContent(content);
  const listItems: React.ReactNode[] = [];
  const elements: React.ReactNode[] = [];
  
  parsedParts.forEach((part, index) => {
    const key = `part-${index}`;
    
    switch (part.type) {
      case 'code':
        // Flush any pending list items
        if (listItems.length > 0) {
          elements.push(
            <ul key={`list-${elements.length}`} className="space-y-1 my-2 ml-4">
              {listItems.map((item, i) => (
                <li key={i} className="flex items-start">
                  <span className="text-muted-foreground mr-2">•</span>
                  {item}
                </li>
              ))}
            </ul>
          );
          listItems.length = 0;
        }
        elements.push(
          <CodeBlock 
            key={key}
            code={part.content} 
            language={part.language}
          />
        );
        break;
        
      case 'inline-code':
        elements.push(renderInlinePart(part, key));
        break;
        
      case 'bold':
        elements.push(renderInlinePart(part, key));
        break;
        
      case 'italic':
        elements.push(renderInlinePart(part, key));
        break;
        
      case 'link':
        elements.push(renderInlinePart(part, key));
        break;
        
      case 'heading':
        const HeadingTag = `h${part.level || 3}` as keyof JSX.IntrinsicElements;
        const headingClasses = {
          1: 'text-2xl font-bold mt-4 mb-2',
          2: 'text-xl font-bold mt-3 mb-2',
          3: 'text-lg font-semibold mt-2 mb-1',
          4: 'text-base font-semibold mt-2 mb-1',
          5: 'text-sm font-semibold mt-1 mb-1',
          6: 'text-sm font-semibold mt-1 mb-1'
        };
        const headingLevel = (part.level || 3) as keyof typeof headingClasses;
        elements.push(
          React.createElement(
            HeadingTag,
            { key, className: headingClasses[headingLevel] },
            part.content
          )
        );
        break;
        
      case 'list':
        // Parse the list item content for markdown
        const listContent = parseMessageContent(part.content);
        const listElements = listContent.map((listPart, i) =>
          renderInlinePart(listPart, `${key}-${i}`)
        );
        
        listItems.push(
          <span key={key}>{listElements}</span>
        );
        break;
        
      case 'linebreak':
        // Flush any pending list items
        if (listItems.length > 0) {
          elements.push(
            <ul key={`list-${elements.length}`} className="space-y-1 my-2 ml-4">
              {listItems.map((item, i) => (
                <li key={i} className="flex items-start">
                  <span className="text-muted-foreground mr-2">•</span>
                  {item}
                </li>
              ))}
            </ul>
          );
          listItems.length = 0;
        }
        elements.push(<br key={key} />);
        break;
        
      case 'text':
        elements.push(part.content);
        break;
    }
  });
  
  // Flush any remaining list items
  if (listItems.length > 0) {
    elements.push(
      <ul key={`list-${elements.length}`} className="space-y-1 my-2 ml-4">
        {listItems.map((item, i) => (
          <li key={i} className="flex items-start">
            <span className="text-muted-foreground mr-2">•</span>
            {item}
          </li>
        ))}
      </ul>
    );
  }
  
  return (
    <div className={cn("break-words overflow-wrap-anywhere", className)}>
      {elements}
    </div>
  );
}
