import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Message } from '../lib/api';
import { useState } from 'react';

interface MessageItemProps {
  message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = async (code: string, id: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCode(id);
      setTimeout(() => setCopiedCode(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : ''}`}>
      <div
        className={`max-w-3xl rounded-lg px-4 py-3 ${
          isUser ? 'bg-blue-600 text-white' : 'bg-gray-800'
        }`}
      >
        <div className={`prose prose-sm max-w-none ${isUser ? 'prose-invert' : 'prose-invert'}`}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              pre({ children, ...props }: any) {
                const codeElement = children?.props;
                if (!codeElement) return <pre {...props}>{children}</pre>;
                
                const className = codeElement.className || '';
                const match = /language-(\w+)/.exec(className);
                
                if (!match) {
                  return <pre {...props}>{children}</pre>;
                }
                
                const language = match[1];
                const codeString = String(codeElement.children).replace(/\n$/, '');
                const codeId = `code-${message.id}-${Math.random()}`;
                
                const languageDisplay = {
                  js: 'JavaScript',
                  javascript: 'JavaScript',
                  py: 'Python',
                  python: 'Python',
                  bash: 'Bash',
                  sh: 'Shell',
                  shell: 'Shell',
                  sql: 'SQL',
                  json: 'JSON',
                  yaml: 'YAML',
                  yml: 'YAML',
                  html: 'HTML',
                  css: 'CSS',
                  typescript: 'TypeScript',
                  ts: 'TypeScript',
                  go: 'Go',
                  rust: 'Rust',
                  java: 'Java',
                  cpp: 'C++',
                  c: 'C',
                  cs: 'C#',
                  php: 'PHP',
                  ruby: 'Ruby',
                  r: 'R',
                  swift: 'Swift',
                  kotlin: 'Kotlin',
                  scala: 'Scala',
                  dockerfile: 'Dockerfile',
                  xml: 'XML',
                  markdown: 'Markdown',
                  md: 'Markdown',
                };
                
                return (
                  <div className="relative group my-4">
                    <div className="bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
                      <div className="bg-gray-800 px-4 py-2 border-b border-gray-700 flex items-center justify-between">
                        <span className="text-xs text-gray-400 font-medium">
                          {(languageDisplay as any)[language.toLowerCase()] || language.toUpperCase()}
                        </span>
                        <button
                          onClick={() => copyToClipboard(codeString, codeId)}
                          className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity text-white"
                        >
                          {copiedCode === codeId ? 'Copied!' : 'Copy'}
                        </button>
                      </div>
                      <div className="overflow-x-auto">
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={language}
                          PreTag="div"
                          customStyle={{
                            margin: 0,
                            padding: '1rem',
                            background: 'transparent',
                          } as React.CSSProperties}
                        >
                          {codeString}
                        </SyntaxHighlighter>
                      </div>
                    </div>
                  </div>
                );
              },
              code({ className, children, ...props }: any) {
                return (
                  <code className="bg-gray-700 px-1 py-0.5 rounded text-sm" {...props}>
                    {children}
                  </code>
                );
              },
              // Customize other elements
              p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc list-inside mb-4">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside mb-4">{children}</ol>,
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-gray-600 pl-4 italic my-4">
                  {children}
                </blockquote>
              ),
              a: ({ href, children }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 underline"
                >
                  {children}
                </a>
              ),
              h1: ({ children }) => <h1 className="text-2xl font-bold mb-4">{children}</h1>,
              h2: ({ children }) => <h2 className="text-xl font-bold mb-3">{children}</h2>,
              h3: ({ children }) => <h3 className="text-lg font-bold mb-2">{children}</h3>,
              table: ({ children }) => (
                <div className="overflow-x-auto my-4">
                  <table className="min-w-full divide-y divide-gray-700">{children}</table>
                </div>
              ),
              th: ({ children }) => (
                <th className="px-4 py-2 bg-gray-800 text-left text-sm font-medium text-gray-300">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="px-4 py-2 text-sm text-gray-300 border-t border-gray-700">
                  {children}
                </td>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        
        {/* Message metadata */}
        {message.model && (
          <div className="mt-2 text-xs text-gray-400">
            {message.model} · {message.tokens_used} tokens · ${message.cost?.toFixed(4)}
          </div>
        )}
      </div>
    </div>
  );
}