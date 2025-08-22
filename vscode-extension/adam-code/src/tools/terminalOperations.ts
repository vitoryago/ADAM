/**
 * Terminal Operations for ADAM VSCode Extension
 * Provides command execution, script running, and process management
 */

import * as vscode from 'vscode';
import { spawn, exec, ExecOptions } from 'child_process';
import * as path from 'path';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface CommandResult {
    success: boolean;
    stdout?: string;
    stderr?: string;
    exitCode?: number;
    message?: string;
}

export interface RunningProcess {
    id: string;
    command: string;
    pid: number;
    startTime: Date;
}

export class TerminalOperations {
    private terminals: Map<string, vscode.Terminal> = new Map();
    private runningProcesses: Map<string, RunningProcess> = new Map();
    
    /**
     * Execute a command and return output
     */
    async executeCommand(command: string, cwd?: string): Promise<CommandResult> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            const execCwd = cwd || workspaceFolder?.uri.fsPath || process.cwd();
            
            const options: ExecOptions = {
                cwd: execCwd,
                maxBuffer: 10 * 1024 * 1024, // 10MB buffer
                timeout: 60000, // 60 second timeout
                env: { ...process.env }
            };
            
            // Show progress notification
            return await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Executing: ${command.substring(0, 50)}...`,
                cancellable: true
            }, async (progress, token) => {
                try {
                    const { stdout, stderr } = await execAsync(command, options);
                    
                    return {
                        success: true,
                        stdout: stdout.toString(),
                        stderr: stderr.toString(),
                        exitCode: 0
                    };
                } catch (error: any) {
                    return {
                        success: false,
                        stdout: error.stdout?.toString() || '',
                        stderr: error.stderr?.toString() || error.message,
                        exitCode: error.code || 1,
                        message: `Command failed: ${error.message}`
                    };
                }
            });
        } catch (error: any) {
            return {
                success: false,
                message: `Failed to execute command: ${error.message}`,
                exitCode: 1
            };
        }
    }
    
    /**
     * Run a command in a new terminal window
     */
    async runInTerminal(command: string, name?: string, cwd?: string): Promise<CommandResult> {
        try {
            const terminalName = name || `ADAM: ${command.substring(0, 20)}...`;
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            const terminalCwd = cwd || workspaceFolder?.uri.fsPath;
            
            // Create or reuse terminal
            let terminal = this.terminals.get(terminalName);
            if (!terminal || terminal.exitStatus) {
                terminal = vscode.window.createTerminal({
                    name: terminalName,
                    cwd: terminalCwd
                });
                this.terminals.set(terminalName, terminal);
            }
            
            // Show terminal and send command
            terminal.show();
            terminal.sendText(command);
            
            return {
                success: true,
                message: `Command running in terminal: ${terminalName}`
            };
        } catch (error: any) {
            return {
                success: false,
                message: `Failed to run in terminal: ${error.message}`
            };
        }
    }
    
    /**
     * Run npm scripts
     */
    async runNpmScript(scriptName: string, args?: string): Promise<CommandResult> {
        const command = `npm run ${scriptName}${args ? ` -- ${args}` : ''}`;
        return this.executeCommand(command);
    }
    
    /**
     * Run tests with automatic detection
     */
    async runTests(testPath?: string, watch?: boolean): Promise<CommandResult> {
        try {
            // Detect test framework
            const packageJson = await this.readPackageJson();
            const scripts = packageJson?.scripts || {};
            
            let testCommand = '';
            
            // Check for test scripts
            if (scripts.test) {
                testCommand = 'npm test';
            } else if (scripts.jest) {
                testCommand = 'npm run jest';
            } else if (scripts.mocha) {
                testCommand = 'npm run mocha';
            } else if (scripts.vitest) {
                testCommand = 'npm run vitest';
            } else {
                // Try common test commands
                const { success } = await this.executeCommand('which jest');
                if (success) {
                    testCommand = 'jest';
                } else {
                    testCommand = 'npm test';
                }
            }
            
            // Add test path if specified
            if (testPath) {
                testCommand += ` ${testPath}`;
            }
            
            // Add watch mode
            if (watch) {
                testCommand += ' --watch';
                return this.runInTerminal(testCommand, 'ADAM Tests (Watch)');
            }
            
            return this.executeCommand(testCommand);
        } catch (error: any) {
            return {
                success: false,
                message: `Failed to run tests: ${error.message}`
            };
        }
    }
    
    /**
     * Install npm packages
     */
    async installPackage(packageName: string, isDev?: boolean, isGlobal?: boolean): Promise<CommandResult> {
        let command = 'npm install';
        
        if (isGlobal) {
            command += ' -g';
        } else if (isDev) {
            command += ' --save-dev';
        }
        
        command += ` ${packageName}`;
        
        return this.executeCommand(command);
    }
    
    /**
     * Uninstall npm packages
     */
    async uninstallPackage(packageName: string, isGlobal?: boolean): Promise<CommandResult> {
        const command = `npm uninstall ${isGlobal ? '-g ' : ''}${packageName}`;
        return this.executeCommand(command);
    }
    
    /**
     * Run build command
     */
    async runBuild(buildCommand?: string): Promise<CommandResult> {
        try {
            // Use provided command or detect from package.json
            if (buildCommand) {
                return this.executeCommand(buildCommand);
            }
            
            const packageJson = await this.readPackageJson();
            const scripts = packageJson?.scripts || {};
            
            if (scripts.build) {
                return this.executeCommand('npm run build');
            } else if (scripts.compile) {
                return this.executeCommand('npm run compile');
            } else if (scripts.dist) {
                return this.executeCommand('npm run dist');
            } else {
                return {
                    success: false,
                    message: 'No build script found in package.json'
                };
            }
        } catch (error: any) {
            return {
                success: false,
                message: `Build failed: ${error.message}`
            };
        }
    }
    
    /**
     * Run linting
     */
    async runLint(fix?: boolean): Promise<CommandResult> {
        try {
            const packageJson = await this.readPackageJson();
            const scripts = packageJson?.scripts || {};
            
            let lintCommand = '';
            
            if (scripts.lint) {
                lintCommand = `npm run lint${fix ? ':fix' : ''}`;
            } else if (scripts.eslint) {
                lintCommand = `npm run eslint${fix ? ' -- --fix' : ''}`;
            } else {
                // Try direct eslint
                lintCommand = `npx eslint .${fix ? ' --fix' : ''}`;
            }
            
            return this.executeCommand(lintCommand);
        } catch (error: any) {
            return {
                success: false,
                message: `Lint failed: ${error.message}`
            };
        }
    }
    
    /**
     * Run type checking
     */
    async runTypeCheck(): Promise<CommandResult> {
        try {
            const packageJson = await this.readPackageJson();
            const scripts = packageJson?.scripts || {};
            
            if (scripts.typecheck) {
                return this.executeCommand('npm run typecheck');
            } else if (scripts['type-check']) {
                return this.executeCommand('npm run type-check');
            } else if (scripts.tsc) {
                return this.executeCommand('npm run tsc');
            } else {
                // Try direct tsc
                return this.executeCommand('npx tsc --noEmit');
            }
        } catch (error: any) {
            return {
                success: false,
                message: `Type check failed: ${error.message}`
            };
        }
    }
    
    /**
     * Start development server
     */
    async startDevServer(): Promise<CommandResult> {
        try {
            const packageJson = await this.readPackageJson();
            const scripts = packageJson?.scripts || {};
            
            let devCommand = '';
            
            if (scripts.dev) {
                devCommand = 'npm run dev';
            } else if (scripts.start) {
                devCommand = 'npm start';
            } else if (scripts.serve) {
                devCommand = 'npm run serve';
            } else if (scripts.watch) {
                devCommand = 'npm run watch';
            } else {
                return {
                    success: false,
                    message: 'No dev server script found'
                };
            }
            
            return this.runInTerminal(devCommand, 'ADAM Dev Server');
        } catch (error: any) {
            return {
                success: false,
                message: `Failed to start dev server: ${error.message}`
            };
        }
    }
    
    /**
     * Kill a running process
     */
    async killProcess(processId: string): Promise<CommandResult> {
        const process = this.runningProcesses.get(processId);
        if (!process) {
            return {
                success: false,
                message: `Process ${processId} not found`
            };
        }
        
        try {
            await execAsync(`kill -9 ${process.pid}`);
            this.runningProcesses.delete(processId);
            
            return {
                success: true,
                message: `Process ${processId} terminated`
            };
        } catch (error: any) {
            return {
                success: false,
                message: `Failed to kill process: ${error.message}`
            };
        }
    }
    
    /**
     * List running processes
     */
    getRunningProcesses(): RunningProcess[] {
        return Array.from(this.runningProcesses.values());
    }
    
    /**
     * Clear all terminals
     */
    clearAllTerminals(): void {
        this.terminals.forEach(terminal => terminal.dispose());
        this.terminals.clear();
    }
    
    /**
     * Read package.json
     */
    private async readPackageJson(): Promise<any> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                return null;
            }
            
            const packageJsonPath = path.join(workspaceFolder.uri.fsPath, 'package.json');
            const doc = await vscode.workspace.openTextDocument(packageJsonPath);
            return JSON.parse(doc.getText());
        } catch {
            return null;
        }
    }
    
    /**
     * Check if command exists
     */
    async commandExists(command: string): Promise<boolean> {
        try {
            const { success } = await this.executeCommand(`which ${command}`);
            return success;
        } catch {
            return false;
        }
    }
    
    /**
     * Get environment variables
     */
    async getEnvironmentVariables(): Promise<Record<string, string>> {
        const result = await this.executeCommand('env');
        const env: Record<string, string> = {};
        
        if (result.success && result.stdout) {
            const lines = result.stdout.split('\n');
            for (const line of lines) {
                const [key, ...valueParts] = line.split('=');
                if (key) {
                    env[key] = valueParts.join('=');
                }
            }
        }
        
        return env;
    }
}