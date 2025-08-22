/**
 * Enhanced ADAM Client with File Operations and Backend Support
 * Combines standalone capabilities with file ops and optional backend connection
 */

import * as vscode from 'vscode';
import { Message } from '../client/adamClient';
import { StandaloneADAMClient } from './standaloneClient';
import { BackendConnector } from '../client/backendConnector';
import { FileOperations } from '../tools/fileOperations';
import { GitOperations } from '../tools/gitOperations';
import { TerminalOperations } from '../tools/terminalOperations';
import { SearchOperations } from '../tools/searchOperations';
import { MultiFileOperations } from '../tools/multiFileOperations';
import { ErrorMemorySystem } from '../memory/errorMemory';
import { ConversationHistoryManager } from '../memory/conversationHistory';

export class EnhancedADAMClient {
    private standaloneClient: StandaloneADAMClient;
    private backendConnector: BackendConnector;
    private fileOps: FileOperations;
    private gitOps: GitOperations;
    private terminalOps: TerminalOperations;
    private searchOps: SearchOperations;
    private multiFileOps: MultiFileOperations;
    private errorMemory: ErrorMemorySystem;
    private conversationHistory: ConversationHistoryManager;
    private useBackend: boolean = false;
    private lastErrorMemoryId?: string;
    
    constructor(private context: vscode.ExtensionContext) {
        // Initialize all components
        this.standaloneClient = new StandaloneADAMClient(context);
        this.backendConnector = new BackendConnector();
        this.fileOps = new FileOperations();
        this.gitOps = new GitOperations();
        this.terminalOps = new TerminalOperations();
        this.searchOps = new SearchOperations();
        this.multiFileOps = new MultiFileOperations();
        this.errorMemory = ErrorMemorySystem.getInstance(context);
        this.conversationHistory = ConversationHistoryManager.getInstance(context);
        
        // Check backend availability
        this.initializeBackend();
    }
    
    /**
     * Initialize backend connection if available
     */
    private async initializeBackend() {
        const config = vscode.workspace.getConfiguration('adam');
        const tryBackend = config.get('tryBackendFirst', true);
        
        if (tryBackend) {
            const connected = await this.backendConnector.testConnection();
            if (connected) {
                this.useBackend = true;
                console.log('ADAM: Connected to backend for full RAG/memory capabilities');
                vscode.window.showInformationMessage('ADAM: Connected to backend with full RAG capabilities');
            } else {
                console.log('ADAM: Backend not available, using standalone mode');
            }
        }
    }
    
    /**
     * Main message handler with command detection
     */
    async sendMessage(content: string, useMemory: boolean = true): Promise<Message> {
        // Track conversation
        const context = {
            activeFile: vscode.window.activeTextEditor?.document.fileName,
            workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
            gitBranch: await this.gitOps.getCurrentBranch()
        };
        
        // Add user message to history
        await this.conversationHistory.addEntry(
            { role: 'user', content, model: 'user' },
            context
        );
        
        // Check for feedback commands
        const feedbackResult = await this.handleFeedback(content);
        if (feedbackResult) {
            return feedbackResult;
        }
        // Check for terminal operation commands
        const terminalOpResult = await this.handleTerminalOperations(content);
        if (terminalOpResult) {
            return terminalOpResult;
        }
        
        // Check for search operation commands
        const searchOpResult = await this.handleSearchOperations(content);
        if (searchOpResult) {
            return searchOpResult;
        }
        
        // Check for file operation commands
        const fileOpResult = await this.handleFileOperations(content);
        if (fileOpResult) {
            return fileOpResult;
        }
        
        // Check for git operation commands
        const gitOpResult = await this.handleGitOperations(content);
        if (gitOpResult) {
            return gitOpResult;
        }
        
        // Use backend if available and connected
        if (this.useBackend) {
            const backendResponse = await this.backendConnector.sendMessageWithRAG(content);
            if (backendResponse) {
                return backendResponse;
            }
            // Fallback to standalone if backend fails
        }
        
        // Use standalone client
        const response = await this.standaloneClient.sendMessage(content, useMemory);
        
        // Track response in history
        await this.conversationHistory.addEntry(
            response,
            context,
            { memoryUsed: useMemory }
        );
        
        return response;
    }
    
    /**
     * Handle file operation commands
     */
    private async handleFileOperations(content: string): Promise<Message | null> {
        const lowerContent = content.toLowerCase();
        
        // Create file command
        if (lowerContent.includes('create file') || lowerContent.includes('create a file') || lowerContent.includes('new file')) {
            const fileMatch = content.match(/(?:file|called|named)\s+["`']?([^\s"`']+)["`']?/i);
            const fileName = fileMatch?.[1];
            
            if (fileName) {
                // Extract content if provided
                const contentMatch = content.match(/```[\w]*\n([\s\S]*?)```/);
                const fileContent = contentMatch?.[1] || '';
                
                const result = await this.fileOps.createFile(fileName, fileContent);
                return {
                    role: 'assistant',
                    content: result.success 
                        ? `✅ ${result.message}\n\nFile created successfully!`
                        : `❌ ${result.message}`,
                    model: 'file-operation'
                };
            }
        }
        
        // Edit file command
        if (lowerContent.includes('edit file') || lowerContent.includes('modify file') || lowerContent.includes('update file')) {
            const fileMatch = content.match(/(?:file|edit|modify)\s+["`']?([^\s"`']+)["`']?/i);
            const fileName = fileMatch?.[1];
            
            if (fileName) {
                // Extract edits from content
                const edits = this.extractEditsFromContent(content);
                
                if (edits.length > 0) {
                    const result = await this.fileOps.editFile(fileName, edits);
                    return {
                        role: 'assistant',
                        content: result.success 
                            ? `✅ ${result.message}`
                            : `❌ ${result.message}`,
                        model: 'file-operation'
                    };
                }
            }
        }
        
        // Delete file command
        if (lowerContent.includes('delete file') || lowerContent.includes('remove file')) {
            const fileMatch = content.match(/(?:delete|remove)\s+["`']?([^\s"`']+)["`']?/i);
            const fileName = fileMatch?.[1];
            
            if (fileName) {
                const result = await this.fileOps.deleteFile(fileName);
                return {
                    role: 'assistant',
                    content: result.success 
                        ? `✅ ${result.message}`
                        : `❌ ${result.message}`,
                    model: 'file-operation'
                };
            }
        }
        
        // List files command
        if (lowerContent.includes('list files') || lowerContent.includes('show files') || lowerContent === 'ls') {
            const dirMatch = content.match(/(?:in|from)\s+["`']?([^\s"`']+)["`']?/i);
            const dir = dirMatch?.[1] || '.';
            
            const result = await this.fileOps.listFiles(dir);
            if (result.success && result.files) {
                return {
                    role: 'assistant',
                    content: `📁 Files in ${dir}:\n\n${result.files.join('\n')}`,
                    model: 'file-operation'
                };
            }
        }
        
        return null;
    }
    
    /**
     * Handle feedback for error learning
     */
    private async handleFeedback(content: string): Promise<Message | null> {
        const lowerContent = content.toLowerCase();
        
        // Check for feedback patterns
        if (lowerContent.includes('that worked') || lowerContent.includes('that fixed it') || 
            lowerContent.includes('solution worked')) {
            // Positive feedback
            const lastError = await this.getLastErrorFromHistory();
            if (lastError) {
                await this.errorMemory.learnFromFeedback(lastError.memoryId, true);
                return {
                    role: 'assistant',
                    content: '✅ Great! I\'ve learned that this solution works for this type of error.',
                    model: 'feedback'
                };
            }
        }
        
        if (lowerContent.includes('didn\'t work') || lowerContent.includes('still broken') || 
            lowerContent.includes('wrong solution')) {
            // Negative feedback
            const lastError = await this.getLastErrorFromHistory();
            if (lastError) {
                // Check if user provided the actual solution
                const solutionMatch = content.match(/(?:actual solution is|correct solution is|should be|try)\s*:?\s*(.+)/i);
                const actualSolution = solutionMatch?.[1];
                
                await this.errorMemory.learnFromFeedback(
                    lastError.memoryId, 
                    false, 
                    actualSolution,
                    content
                );
                
                return {
                    role: 'assistant',
                    content: actualSolution 
                        ? `📝 Thank you! I've learned the correct solution for this error.`
                        : `📝 I've noted that the previous solution didn't work. Could you share what fixed it?`,
                    model: 'feedback'
                };
            }
        }
        
        if (lowerContent.includes('the error was') || lowerContent.includes('i found the error')) {
            // User found an error ADAM missed
            const errorMatch = content.match(/(?:error was|found the error|problem was)\s*:?\s*(.+)/i);
            const errorDescription = errorMatch?.[1];
            
            if (errorDescription) {
                const solutionMatch = content.match(/(?:solution is|fix is|fixed by)\s*:?\s*(.+)/i);
                const solution = solutionMatch?.[1];
                
                const memoryId = await this.errorMemory.rememberError(
                    errorDescription,
                    solution,
                    {
                        file: vscode.window.activeTextEditor?.document.fileName,
                        workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
                    },
                    !!solution
                );
                
                return {
                    role: 'assistant',
                    content: `📚 I've learned about this error${solution ? ' and its solution' : ''}. Thank you for teaching me!`,
                    model: 'feedback'
                };
            }
        }
        
        return null;
    }
    
    /**
     * Handle terminal operation commands
     */
    private async handleTerminalOperations(content: string): Promise<Message | null> {
        const lowerContent = content.toLowerCase();
        
        // Run command
        if (lowerContent.startsWith('run ') || lowerContent.startsWith('execute ') || lowerContent.startsWith('exec ')) {
            const commandMatch = content.match(/(?:run|execute|exec)\s+(.+)/i);
            const command = commandMatch?.[1];
            
            if (command) {
                const startTime = Date.now();
                const result = await this.terminalOps.executeCommand(command);
                const responseTime = Date.now() - startTime;
                
                // If command failed, check error memory for solutions
                let enhancedResponse = '';
                let memoryId: string | undefined;
                
                if (!result.success) {
                    const errorText = result.stderr || result.stdout || result.message || '';
                    
                    // Check error memory for known solution
                    const suggestion = await this.errorMemory.getSuggestedSolution(errorText);
                    if (suggestion) {
                        enhancedResponse = `\n\n💡 **Suggested Solution** (${Math.round(suggestion.confidence * 100)}% confidence):\n${suggestion.solution}`;
                        if (suggestion.commands && suggestion.commands.length > 0) {
                            enhancedResponse += `\n\n**Try running:**\n\`\`\`bash\n${suggestion.commands.join('\n')}\n\`\`\``;
                        }
                    }
                    
                    // Remember this error
                    memoryId = await this.errorMemory.rememberError(
                        errorText,
                        suggestion?.solution,
                        {
                            command,
                            workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
                        },
                        false
                    );
                    
                    // Also analyze with terminal's built-in analyzer
                    const analysis = this.terminalOps.analyzeOutput(errorText);
                    if (analysis.suggestions.length > 0 && !suggestion) {
                        enhancedResponse += `\n\n💡 **Suggestions:**\n${analysis.suggestions.map(s => `• ${s}`).join('\n')}`;
                    }
                }
                
                const response: Message = {
                    role: 'assistant',
                    content: result.success 
                        ? `✅ Command executed:\n\`\`\`\n${result.stdout}\n\`\`\`${result.stderr ? `\n⚠️ Warnings:\n\`\`\`\n${result.stderr}\n\`\`\`` : ''}`
                        : `❌ Command failed: ${result.message}\n${result.stderr ? `\`\`\`\n${result.stderr}\n\`\`\`` : ''}${enhancedResponse}`,
                    model: 'terminal-operation'
                };
                
                // Track in history with error info
                await this.conversationHistory.addEntry(
                    response,
                    {
                        activeFile: vscode.window.activeTextEditor?.document.fileName,
                        workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
                    },
                    { 
                        responseTime,
                        errorHandled: !result.success,
                        memoryUsed: !result.success && enhancedResponse.includes('Suggested Solution')
                    }
                );
                
                // Store memory ID for potential feedback
                if (memoryId) {
                    this.lastErrorMemoryId = memoryId;
                }
                
                return response;
            }
        }
        
        // Install package
        if (lowerContent.includes('install package') || lowerContent.includes('npm install')) {
            const packageMatch = content.match(/(?:install|add)\s+(?:package\s+)?([a-z0-9@\-/]+)/i);
            const packageName = packageMatch?.[1];
            
            if (packageName) {
                const isDev = lowerContent.includes('dev') || lowerContent.includes('--save-dev');
                const result = await this.terminalOps.installPackage(packageName, isDev);
                return {
                    role: 'assistant',
                    content: result.success 
                        ? `✅ Package installed: ${packageName}`
                        : `❌ Failed to install package: ${result.message}`,
                    model: 'terminal-operation'
                };
            }
        }
        
        // Run tests
        if (lowerContent.includes('run test') || lowerContent.includes('test') && !lowerContent.includes('latest')) {
            const watch = lowerContent.includes('watch');
            const result = await this.terminalOps.runTests(undefined, watch);
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ Tests ${watch ? 'running in watch mode' : 'completed'}:\n\`\`\`\n${result.stdout}\n\`\`\``
                    : `❌ Tests failed: ${result.message}`,
                model: 'terminal-operation'
            };
        }
        
        // Run build
        if (lowerContent.includes('build') || lowerContent.includes('compile')) {
            const result = await this.terminalOps.runBuild();
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ Build completed:\n\`\`\`\n${result.stdout}\n\`\`\``
                    : `❌ Build failed: ${result.message}`,
                model: 'terminal-operation'
            };
        }
        
        // Run lint
        if (lowerContent.includes('lint') || lowerContent.includes('eslint')) {
            const fix = lowerContent.includes('fix');
            const result = await this.terminalOps.runLint(fix);
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ Lint ${fix ? 'fixed issues' : 'completed'}:\n\`\`\`\n${result.stdout}\n\`\`\``
                    : `❌ Lint failed: ${result.message}`,
                model: 'terminal-operation'
            };
        }
        
        // Type check
        if (lowerContent.includes('typecheck') || lowerContent.includes('type check') || lowerContent.includes('tsc')) {
            const result = await this.terminalOps.runTypeCheck();
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ Type check passed:\n\`\`\`\n${result.stdout}\n\`\`\``
                    : `❌ Type check failed:\n\`\`\`\n${result.stderr || result.stdout}\n\`\`\``,
                model: 'terminal-operation'
            };
        }
        
        // Start dev server
        if (lowerContent.includes('start dev') || lowerContent.includes('dev server') || lowerContent === 'npm start') {
            const result = await this.terminalOps.startDevServer();
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ Dev server started in terminal`
                    : `❌ Failed to start dev server: ${result.message}`,
                model: 'terminal-operation'
            };
        }
        
        return null;
    }
    
    /**
     * Handle search operation commands
     */
    private async handleSearchOperations(content: string): Promise<Message | null> {
        const lowerContent = content.toLowerCase();
        
        // Search in files
        if (lowerContent.includes('search for') || lowerContent.includes('find') || lowerContent.includes('grep')) {
            const searchMatch = content.match(/(?:search for|find|grep)\s+["`']?([^"`'\n]+)["`']?/i);
            const pattern = searchMatch?.[1];
            
            if (pattern) {
                const results = await this.searchOps.searchInFiles({ 
                    pattern,
                    maxResults: 20,
                    caseSensitive: false
                });
                
                if (results.length === 0) {
                    return {
                        role: 'assistant',
                        content: `No matches found for: "${pattern}"`,
                        model: 'search-operation'
                    };
                }
                
                const formatted = results.map(r => 
                    `📄 ${r.file}:${r.line}:${r.column}\n   ${r.preview.trim()}`
                ).join('\n\n');
                
                return {
                    role: 'assistant',
                    content: `🔍 Found ${results.length} matches:\n\n${formatted}`,
                    model: 'search-operation'
                };
            }
        }
        
        // Find TODOs
        if (lowerContent.includes('find todo') || lowerContent.includes('show todo')) {
            const results = await this.searchOps.findTodos();
            
            if (results.length === 0) {
                return {
                    role: 'assistant',
                    content: 'No TODOs found in the codebase',
                    model: 'search-operation'
                };
            }
            
            const formatted = results.map(r => 
                `📝 ${r.file}:${r.line} - ${r.preview.trim()}`
            ).join('\n');
            
            return {
                role: 'assistant',
                content: `📋 Found ${results.length} TODOs:\n\n${formatted}`,
                model: 'search-operation'
            };
        }
        
        // Find files
        if (lowerContent.includes('find file') || lowerContent.includes('locate file')) {
            const fileMatch = content.match(/(?:find|locate)\s+file\s+["`']?([^"`'\n]+)["`']?/i);
            const pattern = fileMatch?.[1];
            
            if (pattern) {
                const files = await this.searchOps.findFiles(`**/*${pattern}*`);
                
                if (files.length === 0) {
                    return {
                        role: 'assistant',
                        content: `No files found matching: "${pattern}"`,
                        model: 'search-operation'
                    };
                }
                
                return {
                    role: 'assistant',
                    content: `📁 Found ${files.length} files:\n\n${files.map(f => `  📄 ${f}`).join('\n')}`,
                    model: 'search-operation'
                };
            }
        }
        
        // Find symbol
        if (lowerContent.includes('find symbol') || lowerContent.includes('find function') || lowerContent.includes('find class')) {
            const symbolMatch = content.match(/(?:symbol|function|class)\s+["`']?([^"`'\n]+)["`']?/i);
            const symbolName = symbolMatch?.[1];
            
            if (symbolName) {
                const symbols = await this.searchOps.findSymbol(symbolName);
                
                if (symbols.length === 0) {
                    return {
                        role: 'assistant',
                        content: `No symbols found matching: "${symbolName}"`,
                        model: 'search-operation'
                    };
                }
                
                const formatted = symbols.map(s => {
                    const path = vscode.workspace.asRelativePath(s.location.uri);
                    return `  ${s.kind === vscode.SymbolKind.Function ? '🔧' : '📦'} ${s.name} - ${path}:${s.location.range.start.line + 1}`;
                }).join('\n');
                
                return {
                    role: 'assistant',
                    content: `🔍 Found ${symbols.length} symbols:\n\n${formatted}`,
                    model: 'search-operation'
                };
            }
        }
        
        // Replace in files
        if (lowerContent.includes('replace') && lowerContent.includes('with')) {
            const replaceMatch = content.match(/replace\s+["`']([^"`']+)["`']\s+with\s+["`']([^"`']+)["`']/i);
            
            if (replaceMatch) {
                const [, search, replacement] = replaceMatch;
                const result = await this.searchOps.replaceInFiles(search, replacement);
                
                return {
                    role: 'assistant',
                    content: `✅ Replaced ${result.occurrencesReplaced} occurrences in ${result.filesChanged} files`,
                    model: 'search-operation'
                };
            }
        }
        
        return null;
    }
    
    /**
     * Handle git operation commands
     */
    private async handleGitOperations(content: string): Promise<Message | null> {
        const lowerContent = content.toLowerCase();
        
        // Git status
        if (lowerContent.includes('git status') || lowerContent === 'status') {
            const result = await this.gitOps.status();
            return {
                role: 'assistant',
                content: result.success 
                    ? `📊 Git Status:\n\`\`\`\n${result.status}\n\`\`\``
                    : `❌ ${result.message}`,
                model: 'git-operation'
            };
        }
        
        // Git commit
        if (lowerContent.includes('commit') && !lowerContent.includes('uncommit')) {
            const messageMatch = content.match(/(?:message|commit)\s+["`']([^"`']+)["`']/i);
            const commitMessage = messageMatch?.[1] || 'Update from ADAM';
            
            const addAll = lowerContent.includes('all') || lowerContent.includes('-a');
            const result = await this.gitOps.commit(commitMessage, addAll);
            
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ ${result.message}`
                    : `❌ ${result.message}`,
                model: 'git-operation'
            };
        }
        
        // Git push
        if (lowerContent.includes('git push') || lowerContent === 'push') {
            const force = lowerContent.includes('force');
            const result = await this.gitOps.push(force);
            
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ ${result.message}`
                    : `❌ ${result.message}`,
                model: 'git-operation'
            };
        }
        
        // Git pull
        if (lowerContent.includes('git pull') || lowerContent === 'pull') {
            const result = await this.gitOps.pull();
            
            return {
                role: 'assistant',
                content: result.success 
                    ? `✅ ${result.message}`
                    : `❌ ${result.message}`,
                model: 'git-operation'
            };
        }
        
        // Create branch
        if (lowerContent.includes('create branch') || lowerContent.includes('new branch')) {
            const branchMatch = content.match(/(?:branch|called|named)\s+["`']?([^\s"`']+)["`']?/i);
            const branchName = branchMatch?.[1];
            
            if (branchName) {
                const result = await this.gitOps.createBranch(branchName);
                return {
                    role: 'assistant',
                    content: result.success 
                        ? `✅ ${result.message}`
                        : `❌ ${result.message}`,
                    model: 'git-operation'
                };
            }
        }
        
        // List branches
        if (lowerContent.includes('list branches') || lowerContent.includes('show branches')) {
            const result = await this.gitOps.listBranches();
            
            if (result.success && result.branches) {
                const current = await this.gitOps.getCurrentBranch();
                const branchList = result.branches.map(b => 
                    b === current ? `→ ${b} (current)` : `  ${b}`
                ).join('\n');
                
                return {
                    role: 'assistant',
                    content: `🌿 Git Branches:\n\`\`\`\n${branchList}\n\`\`\``,
                    model: 'git-operation'
                };
            }
        }
        
        // Git log
        if (lowerContent.includes('git log') || lowerContent.includes('show commits')) {
            const limitMatch = content.match(/(?:last|recent)\s+(\d+)/i);
            const limit = limitMatch ? parseInt(limitMatch[1]) : 10;
            
            const result = await this.gitOps.log(limit);
            
            if (result.success && result.commits) {
                return {
                    role: 'assistant',
                    content: `📜 Recent Commits:\n\`\`\`\n${result.commits.join('\n')}\n\`\`\``,
                    model: 'git-operation'
                };
            }
        }
        
        return null;
    }
    
    /**
     * Extract edit instructions from content
     */
    private extractEditsFromContent(content: string): Array<{ oldText: string; newText: string }> {
        const edits: Array<{ oldText: string; newText: string }> = [];
        
        // Look for replace patterns
        const replaceMatches = content.matchAll(/replace\s+["`']([^"`']+)["`']\s+with\s+["`']([^"`']+)["`']/gi);
        for (const match of replaceMatches) {
            edits.push({ oldText: match[1], newText: match[2] });
        }
        
        // Look for old/new code blocks
        const oldMatch = content.match(/```old\n([\s\S]*?)```/);
        const newMatch = content.match(/```new\n([\s\S]*?)```/);
        
        if (oldMatch && newMatch) {
            edits.push({ oldText: oldMatch[1], newText: newMatch[1] });
        }
        
        return edits;
    }
    
    /**
     * Switch between backend and standalone modes
     */
    async switchMode(useBackend: boolean): Promise<void> {
        if (useBackend) {
            const connected = await this.backendConnector.testConnection();
            if (connected) {
                this.useBackend = true;
                vscode.window.showInformationMessage('Switched to backend mode with full RAG');
            } else {
                vscode.window.showErrorMessage('Backend not available, staying in standalone mode');
            }
        } else {
            this.useBackend = false;
            vscode.window.showInformationMessage('Switched to standalone mode');
        }
    }
    
    /**
     * Get current mode
     */
    getCurrentMode(): string {
        return this.useBackend ? 'backend' : 'standalone';
    }
    
    /**
     * Get memory statistics
     */
    async getMemoryStats(): Promise<any> {
        if (this.useBackend) {
            return await this.backendConnector.getMemoryStats();
        }
        // Return basic stats for standalone
        return {
            mode: 'standalone',
            memories: 'Local storage'
        };
    }
    
    /**
     * Explain code using AI
     */
    async explainCode(code: string, language: string, fileName: string): Promise<string> {
        const prompt = `Explain this ${language} code from ${fileName}:\n\`\`\`${language}\n${code}\n\`\`\``;
        const response = await this.sendMessage(prompt);
        return response.content;
    }
    
    /**
     * Optimize SQL query
     */
    async optimizeSQL(query: string, dialect: string): Promise<{ query: string; explanation: string; improvements: string }> {
        const prompt = `Optimize this ${dialect} SQL query:\n\`\`\`sql\n${query}\n\`\`\`\n\nProvide the optimized query, explanation, and improvements.`;
        const response = await this.sendMessage(prompt);
        
        // Parse response to extract parts
        const optimizedQuery = response.content.match(/```sql\n([\s\S]*?)```/)?.[1] || query;
        
        return {
            query: optimizedQuery,
            explanation: 'Query optimized using AI analysis',
            improvements: response.content
        };
    }
    
    /**
     * Create a git branch with AI-suggested name
     */
    async createBranch(description: string): Promise<{ formattedName: string; description: string }> {
        const formattedName = description
            .toLowerCase()
            .replace(/\s+/g, '-')
            .replace(/[^a-z0-9-]/g, '');
        
        await this.gitOps.createBranch(formattedName);
        
        return {
            formattedName,
            description: `Branch for: ${description}`
        };
    }
    
    /**
     * Generate PR details using AI
     */
    async generatePRDetails(changes: string): Promise<{ title: string; body: string }> {
        const prompt = `Generate a pull request title and description for these changes:\n${changes}`;
        const response = await this.sendMessage(prompt);
        
        // Extract title and body from response
        const lines = response.content.split('\n');
        const title = lines[0].replace(/^#+\s*/, '') || 'Update from ADAM';
        const body = lines.slice(1).join('\n') || response.content;
        
        return { title, body };
    }
    
    /**
     * Generate dbt model
     */
    async generateDBTModel(modelName: string, sourceTable: string): Promise<{ sql: string; documentation: string }> {
        const prompt = `Generate a dbt model named ${modelName} from source table ${sourceTable}. Include SQL and documentation.`;
        const response = await this.sendMessage(prompt);
        
        const sql = response.content.match(/```sql\n([\s\S]*?)```/)?.[1] || 
                   `select * from {{ source('${sourceTable.split('.')[0]}', '${sourceTable.split('.')[1]}') }}`;
        
        return {
            sql,
            documentation: response.content
        };
    }
    
    /**
     * Analyze data pattern
     */
    async analyzeDataPattern(data: string): Promise<{ summary: string; patterns: string; recommendations: string }> {
        const prompt = `Analyze this data for patterns:\n${data}`;
        const response = await this.sendMessage(prompt);
        
        return {
            summary: 'Data analysis complete',
            patterns: response.content,
            recommendations: 'See analysis above'
        };
    }
    
    /**
     * Get last error from history
     */
    private async getLastErrorFromHistory(): Promise<{ memoryId: string } | null> {
        if (this.lastErrorMemoryId) {
            return { memoryId: this.lastErrorMemoryId };
        }
        return null;
    }
    
    /**
     * Get conversation history
     */
    getConversationHistory(limit?: number): any[] {
        return this.conversationHistory.getHistory({ limit });
    }
    
    /**
     * Export conversation history
     */
    exportHistory(format: 'json' | 'markdown' = 'json'): string {
        return this.conversationHistory.exportHistory(format);
    }
    
    /**
     * Get error memory statistics
     */
    getErrorStats(): any {
        return this.errorMemory.getStatistics();
    }
    
    /**
     * Search error memory
     */
    async searchErrorMemory(query: string): Promise<any[]> {
        return this.errorMemory.searchSimilarErrors(query);
    }
    
    /**
     * Add custom error solution
     */
    async teachErrorSolution(error: string, solution: string, explanation: string): Promise<void> {
        const memoryId = await this.errorMemory.rememberError(
            error,
            solution,
            {
                workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
            },
            true
        );
        this.lastErrorMemoryId = memoryId;
    }
    
    /**
     * Clear old conversation history
     */
    clearOldHistory(daysToKeep: number = 30): number {
        return this.conversationHistory.clearOldHistory(daysToKeep);
    }
    
    /**
     * Get conversation statistics
     */
    getConversationStats(): any {
        return this.conversationHistory.getStatistics();
    }
    
    /**
     * Start new conversation session
     */
    startNewSession(title?: string): void {
        this.conversationHistory.startNewSession(title);
        this.lastErrorMemoryId = undefined;
    }
    
    /**
     * Disconnect and cleanup
     */
    disconnect(): void {
        // End conversation session
        this.conversationHistory.endCurrentSession();
        
        // Cleanup if needed
        console.log('ADAM Enhanced Client disconnected');
    }
}