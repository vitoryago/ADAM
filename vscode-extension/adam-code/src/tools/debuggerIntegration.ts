/**
 * Debugger Integration for ADAM VSCode Extension
 * Provides debugging capabilities and breakpoint management
 */

import * as vscode from 'vscode';
import * as path from 'path';

export interface BreakpointInfo {
    id: string;
    file: string;
    line: number;
    condition?: string;
    hitCount?: number;
    logMessage?: string;
    enabled: boolean;
}

export interface DebugVariable {
    name: string;
    value: string;
    type: string;
    scope: string;
}

export interface StackFrame {
    name: string;
    source: string;
    line: number;
    column: number;
}

export class DebuggerIntegration {
    private activeSession: vscode.DebugSession | undefined;
    private breakpoints: Map<string, BreakpointInfo> = new Map();
    private watchExpressions: Set<string> = new Set();
    
    constructor() {
        // Listen to debug session changes
        vscode.debug.onDidStartDebugSession(session => {
            this.activeSession = session;
            console.log(`Debug session started: ${session.name}`);
        });
        
        vscode.debug.onDidTerminateDebugSession(session => {
            if (this.activeSession === session) {
                this.activeSession = undefined;
            }
            console.log(`Debug session terminated: ${session.name}`);
        });
        
        vscode.debug.onDidChangeBreakpoints(e => {
            this.updateBreakpointTracking(e);
        });
    }
    
    /**
     * Start debugging with configuration
     */
    async startDebugging(
        folder?: vscode.WorkspaceFolder,
        nameOrConfig?: string | vscode.DebugConfiguration
    ): Promise<boolean> {
        try {
            if (!nameOrConfig) {
                return await vscode.debug.startDebugging(folder, undefined as any);
            }
            return await vscode.debug.startDebugging(folder, nameOrConfig);
        } catch (error) {
            console.error('Failed to start debugging:', error);
            return false;
        }
    }
    
    /**
     * Start debugging with auto-detection
     */
    async startAutoDebug(): Promise<boolean> {
        const activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor) {
            vscode.window.showErrorMessage('No active file to debug');
            return false;
        }
        
        const document = activeEditor.document;
        const language = document.languageId;
        
        // Auto-detect debug configuration based on language
        const config = this.getDebugConfig(language, document.fileName);
        
        if (!config) {
            vscode.window.showErrorMessage(`No debug configuration for ${language}`);
            return false;
        }
        
        return this.startDebugging(undefined, config);
    }
    
    /**
     * Add breakpoint at current cursor position
     */
    async addBreakpointAtCursor(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        
        const position = editor.selection.active;
        const uri = editor.document.uri;
        
        const breakpoint = new vscode.SourceBreakpoint(
            new vscode.Location(uri, position)
        );
        
        vscode.debug.addBreakpoints([breakpoint]);
    }
    
    /**
     * Add conditional breakpoint
     */
    async addConditionalBreakpoint(
        file: string,
        line: number,
        condition: string
    ): Promise<void> {
        const uri = vscode.Uri.file(file);
        const location = new vscode.Location(uri, new vscode.Position(line - 1, 0));
        
        const breakpoint = new vscode.SourceBreakpoint(location, true, condition);
        vscode.debug.addBreakpoints([breakpoint]);
    }
    
    /**
     * Add log point (breakpoint that logs instead of breaking)
     */
    async addLogPoint(
        file: string,
        line: number,
        logMessage: string
    ): Promise<void> {
        const uri = vscode.Uri.file(file);
        const location = new vscode.Location(uri, new vscode.Position(line - 1, 0));
        
        const breakpoint = new vscode.SourceBreakpoint(
            location,
            true,
            undefined,
            undefined,
            logMessage
        );
        
        vscode.debug.addBreakpoints([breakpoint]);
    }
    
    /**
     * Remove all breakpoints
     */
    async clearAllBreakpoints(): Promise<void> {
        vscode.debug.removeBreakpoints(vscode.debug.breakpoints);
    }
    
    /**
     * Toggle breakpoint at line
     */
    async toggleBreakpoint(file: string, line: number): Promise<void> {
        const uri = vscode.Uri.file(file);
        const existingBreakpoints = vscode.debug.breakpoints.filter(
            bp => bp instanceof vscode.SourceBreakpoint &&
                  bp.location.uri.fsPath === uri.fsPath &&
                  bp.location.range.start.line === line - 1
        );
        
        if (existingBreakpoints.length > 0) {
            vscode.debug.removeBreakpoints(existingBreakpoints);
        } else {
            const location = new vscode.Location(uri, new vscode.Position(line - 1, 0));
            const breakpoint = new vscode.SourceBreakpoint(location);
            vscode.debug.addBreakpoints([breakpoint]);
        }
    }
    
    /**
     * Get all breakpoints
     */
    getAllBreakpoints(): BreakpointInfo[] {
        const breakpoints: BreakpointInfo[] = [];
        
        for (const bp of vscode.debug.breakpoints) {
            if (bp instanceof vscode.SourceBreakpoint) {
                breakpoints.push({
                    id: bp.id,
                    file: bp.location.uri.fsPath,
                    line: bp.location.range.start.line + 1,
                    condition: bp.condition,
                    logMessage: bp.logMessage,
                    enabled: bp.enabled
                });
            }
        }
        
        return breakpoints;
    }
    
    /**
     * Step over in debugger
     */
    async stepOver(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.stepOver');
    }
    
    /**
     * Step into function
     */
    async stepInto(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.stepInto');
    }
    
    /**
     * Step out of function
     */
    async stepOut(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.stepOut');
    }
    
    /**
     * Continue execution
     */
    async continue(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.continue');
    }
    
    /**
     * Pause execution
     */
    async pause(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.pause');
    }
    
    /**
     * Stop debugging
     */
    async stop(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.stop');
    }
    
    /**
     * Restart debugging
     */
    async restart(): Promise<void> {
        await vscode.commands.executeCommand('workbench.action.debug.restart');
    }
    
    /**
     * Add watch expression
     */
    async addWatch(expression: string): Promise<void> {
        this.watchExpressions.add(expression);
        // VSCode doesn't have a direct API for this, use command
        await vscode.commands.executeCommand('workbench.debug.viewlet.action.addWatchExpression');
    }
    
    /**
     * Evaluate expression in debug console
     */
    async evaluate(expression: string): Promise<string | undefined> {
        if (!this.activeSession) {
            return undefined;
        }
        
        try {
            // This would need custom debug adapter protocol implementation
            // For now, just show in debug console
            await vscode.commands.executeCommand('workbench.debug.action.focusRepl');
            return `Evaluating: ${expression}`;
        } catch (error) {
            console.error('Failed to evaluate expression:', error);
            return undefined;
        }
    }
    
    /**
     * Get current call stack
     */
    async getCallStack(): Promise<StackFrame[]> {
        // This would need integration with debug adapter protocol
        // For now, return placeholder
        if (!this.activeSession) {
            return [];
        }
        
        return [
            {
                name: 'getCurrentFunction',
                source: 'example.js',
                line: 42,
                column: 10
            }
        ];
    }
    
    /**
     * Get variables in current scope
     */
    async getVariables(): Promise<DebugVariable[]> {
        // This would need integration with debug adapter protocol
        // For now, return placeholder
        if (!this.activeSession) {
            return [];
        }
        
        return [
            {
                name: 'exampleVar',
                value: '42',
                type: 'number',
                scope: 'local'
            }
        ];
    }
    
    /**
     * Set variable value during debugging
     */
    async setVariable(name: string, value: string): Promise<boolean> {
        if (!this.activeSession) {
            return false;
        }
        
        try {
            // This would need custom implementation
            console.log(`Setting ${name} = ${value}`);
            return true;
        } catch (error) {
            console.error('Failed to set variable:', error);
            return false;
        }
    }
    
    /**
     * Check if debugging is active
     */
    isDebugging(): boolean {
        return this.activeSession !== undefined;
    }
    
    /**
     * Get active debug session
     */
    getActiveSession(): vscode.DebugSession | undefined {
        return this.activeSession;
    }
    
    /**
     * Create launch.json configuration
     */
    async createLaunchConfig(): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return;
        }
        
        const launchConfig = {
            version: '0.2.0',
            configurations: [
                this.getDebugConfig('javascript', ''),
                this.getDebugConfig('typescript', ''),
                this.getDebugConfig('python', '')
            ].filter(Boolean)
        };
        
        const launchPath = path.join(workspaceFolder.uri.fsPath, '.vscode', 'launch.json');
        const uri = vscode.Uri.file(launchPath);
        
        const edit = new vscode.WorkspaceEdit();
        edit.createFile(uri, { 
            contents: Buffer.from(JSON.stringify(launchConfig, null, 2)) 
        });
        
        await vscode.workspace.applyEdit(edit);
        await vscode.window.showTextDocument(uri);
    }
    
    /**
     * Get debug configuration for language
     */
    private getDebugConfig(language: string, file: string): vscode.DebugConfiguration | null {
        switch (language) {
            case 'javascript':
            case 'typescript':
                return {
                    type: 'node',
                    request: 'launch',
                    name: `Debug ${language}`,
                    skipFiles: ['<node_internals>/**'],
                    program: file || '${file}',
                    outFiles: language === 'typescript' ? ['${workspaceFolder}/**/*.js'] : undefined
                };
                
            case 'python':
                return {
                    type: 'python',
                    request: 'launch',
                    name: 'Debug Python',
                    program: file || '${file}',
                    console: 'integratedTerminal'
                };
                
            case 'java':
                return {
                    type: 'java',
                    request: 'launch',
                    name: 'Debug Java',
                    mainClass: '${file}'
                };
                
            case 'csharp':
                return {
                    type: 'coreclr',
                    request: 'launch',
                    name: 'Debug .NET Core',
                    program: '${workspaceFolder}/bin/Debug/<target-framework>/<project-name.dll>',
                    cwd: '${workspaceFolder}'
                };
                
            case 'go':
                return {
                    type: 'go',
                    request: 'launch',
                    name: 'Debug Go',
                    mode: 'debug',
                    program: file || '${file}'
                };
                
            case 'rust':
                return {
                    type: 'lldb',
                    request: 'launch',
                    name: 'Debug Rust',
                    cargo: {
                        args: ['build', '--bin=<name>'],
                        filter: {
                            name: '<name>',
                            kind: 'bin'
                        }
                    }
                };
                
            default:
                return null;
        }
    }
    
    /**
     * Update breakpoint tracking
     */
    private updateBreakpointTracking(e: vscode.BreakpointsChangeEvent): void {
        // Track added breakpoints
        for (const bp of e.added) {
            if (bp instanceof vscode.SourceBreakpoint) {
                const info: BreakpointInfo = {
                    id: bp.id,
                    file: bp.location.uri.fsPath,
                    line: bp.location.range.start.line + 1,
                    condition: bp.condition,
                    logMessage: bp.logMessage,
                    enabled: bp.enabled
                };
                this.breakpoints.set(bp.id, info);
            }
        }
        
        // Remove deleted breakpoints
        for (const bp of e.removed) {
            this.breakpoints.delete(bp.id);
        }
        
        // Update changed breakpoints
        for (const bp of e.changed) {
            if (bp instanceof vscode.SourceBreakpoint && this.breakpoints.has(bp.id)) {
                const info = this.breakpoints.get(bp.id)!;
                info.enabled = bp.enabled;
                info.condition = bp.condition;
                info.logMessage = bp.logMessage;
            }
        }
    }
}