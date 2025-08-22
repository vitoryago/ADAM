import * as vscode from 'vscode';
import { ADAMChatProvider } from './providers/chatProvider';
import { ADAMClient } from './client/adamClient';
import { StandaloneADAMClient } from './standalone/standaloneClient';
import { EnhancedADAMClient } from './standalone/enhancedClient';
import { GitIntegration } from './integrations/gitIntegration';
import { SQLOptimizer } from './tools/sqlOptimizer';
import { DBTGenerator } from './tools/dbtGenerator';
import { FileManager } from './tools/fileManager';
import { VoiceChat } from './features/voiceChat';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let adamClient: ADAMClient | StandaloneADAMClient | EnhancedADAMClient;
let chatProvider: ADAMChatProvider;
let backendProcess: ChildProcess | null = null;

export async function activate(context: vscode.ExtensionContext) {
    console.log('ADAM Code is activating...');
    
    try {
        // Check if we should use standalone mode (default: true)
        const config = vscode.workspace.getConfiguration('adam');
        const useStandalone = config.get('standalone', true);
        
        if (useStandalone) {
            // Use enhanced standalone ADAM with file operations and optional backend
            adamClient = new EnhancedADAMClient(context);
            console.log('ADAM running in enhanced mode with file operations!');
        } else {
            // Use backend mode - start backend automatically
            const backendStarted = await startBackend(context);
            if (!backendStarted) {
                // Fallback to enhanced standalone if backend fails
                vscode.window.showWarningMessage('Failed to start ADAM backend. Using enhanced standalone mode.');
                adamClient = new EnhancedADAMClient(context);
            } else {
                adamClient = new ADAMClient(
                    config.get('serverUrl') || 'http://localhost:8000',
                    config.get('projectId') || '3a859e97-16fd-46c6-b018-1ede9fade704'
                );
            }
        }

        // Initialize chat provider - cast to ADAMClient for now
        chatProvider = new ADAMChatProvider(context.extensionUri, adamClient as any);

        // Register webview provider BEFORE registering commands
        context.subscriptions.push(
            vscode.window.registerWebviewViewProvider('adam.chatView', chatProvider)
        );

        // Register commands - this must come after provider registration
        registerCommands(context);

        // Initialize features - cast to ADAMClient for compatibility
        const gitIntegration = new GitIntegration();
        const sqlOptimizer = new SQLOptimizer(adamClient as any);
        const dbtGenerator = new DBTGenerator(adamClient as any);
        const fileManager = new FileManager();
        const voiceChat = new VoiceChat(adamClient as any);

        // Status bar item
        const statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        statusBarItem.text = '$(circuit-board) ADAM Ready';
        statusBarItem.tooltip = 'ADAM is connected and ready';
        statusBarItem.command = 'adam.chat';
        statusBarItem.show();
        context.subscriptions.push(statusBarItem);

        console.log('ADAM Code is ready!');
        
    } catch (error) {
        console.error('Failed to activate ADAM Code:', error);
        vscode.window.showErrorMessage(`Failed to activate ADAM: ${error}`);
    }
}

function registerCommands(context: vscode.ExtensionContext) {
    // Main chat command
    const chatCommand = vscode.commands.registerCommand('adam.chat', async () => {
        try {
            // First ensure the sidebar is visible
            await vscode.commands.executeCommand('workbench.view.extension.adam-container');
            // Then focus on the chat view
            await vscode.commands.executeCommand('adam.chatView.focus');
        } catch (error) {
            console.error('Error executing adam.chat command:', error);
            vscode.window.showErrorMessage(`Failed to open ADAM chat: ${error}`);
        }
    });
    context.subscriptions.push(chatCommand);

    // Explain selected code
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return;
            }

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showInformationMessage('Please select code to explain');
                return;
            }

            const language = editor.document.languageId;
            const fileName = editor.document.fileName;

            // Show progress
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is analyzing your code...",
                cancellable: false
            }, async (progress) => {
                try {
                    const explanation = await adamClient.explainCode(selection, language, fileName);
                    
                    // Show explanation in chat
                    chatProvider.addMessage({
                        role: 'user',
                        content: `Explain this ${language} code:\n\`\`\`${language}\n${selection}\n\`\`\``
                    });
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: explanation
                    });
                    chatProvider.show();
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    // Optimize SQL query
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.optimize', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sql') {
                vscode.window.showInformationMessage('Please open a SQL file');
                return;
            }

            const query = editor.document.getText();
            const dialect = vscode.workspace.getConfiguration('adam').get('sqlDialect') as string;

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is optimizing your SQL query...",
                cancellable: false
            }, async (progress) => {
                try {
                    const optimized = await adamClient.optimizeSQL(query, dialect);
                    
                    // Create diff view
                    const uri = vscode.Uri.parse(`adam-optimized:${editor.document.fileName}.optimized.sql`);
                    const doc = await vscode.workspace.openTextDocument(uri);
                    await vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.Beside });
                    
                    // Apply the optimized query
                    const edit = new vscode.WorkspaceEdit();
                    edit.replace(
                        editor.document.uri,
                        new vscode.Range(0, 0, editor.document.lineCount, 0),
                        optimized.query
                    );
                    
                    // Show optimization details
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: `## SQL Optimization Results\n\n${optimized.explanation}\n\n### Performance Improvements:\n${optimized.improvements}`
                    });
                    chatProvider.show();
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    // Create feature branch
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.createBranch', async () => {
            const branchName = await vscode.window.showInputBox({
                prompt: 'Enter branch name (ADAM will help format it)',
                placeHolder: 'feature/add-customer-metrics'
            });

            if (!branchName) {
                return;
            }

            try {
                const result = await adamClient.createBranch(branchName);
                
                // Execute git commands
                const terminal = vscode.window.createTerminal('ADAM Git');
                terminal.show();
                terminal.sendText(`git checkout -b ${result.formattedName}`);
                
                vscode.window.showInformationMessage(`Branch created: ${result.formattedName}`);
                
                // Add to memory
                chatProvider.addMessage({
                    role: 'assistant',
                    content: `Created branch: \`${result.formattedName}\`\n\nReady to start working on: ${result.description}`
                });
            } catch (error) {
                vscode.window.showErrorMessage(`ADAM Error: ${error}`);
            }
        })
    );

    // Create pull request
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.createPR', async () => {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is analyzing changes and creating PR...",
                cancellable: false
            }, async (progress) => {
                try {
                    // Get current changes
                    const gitExtension = vscode.extensions.getExtension('vscode.git')?.exports;
                    const api = gitExtension.getAPI(1);
                    const repo = api.repositories[0];
                    
                    const changes = await repo.diff();
                    const prDetails = await adamClient.generatePRDetails(changes);
                    
                    // Create PR using gh CLI
                    const terminal = vscode.window.createTerminal('ADAM PR');
                    terminal.show();
                    terminal.sendText(`gh pr create --title "${prDetails.title}" --body "${prDetails.body}"`);
                    
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: `## Pull Request Created\n\n**Title:** ${prDetails.title}\n\n**Description:**\n${prDetails.body}`
                    });
                    chatProvider.show();
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    // Generate dbt model
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.dbtGenerate', async () => {
            const modelName = await vscode.window.showInputBox({
                prompt: 'Enter dbt model name',
                placeHolder: 'stg_customers'
            });

            if (!modelName) {
                return;
            }

            const sourceTable = await vscode.window.showInputBox({
                prompt: 'Enter source table',
                placeHolder: 'raw.customers'
            });

            if (!sourceTable) {
                return;
            }

            try {
                const model = await adamClient.generateDBTModel(modelName, sourceTable);
                
                // Create model file
                const dbtProject = vscode.workspace.getConfiguration('adam').get('dbtProject') as string;
                const modelPath = `${dbtProject}/models/staging/${modelName}.sql`;
                
                const uri = vscode.Uri.file(modelPath);
                const edit = new vscode.WorkspaceEdit();
                edit.createFile(uri, { contents: Buffer.from(model.sql) });
                await vscode.workspace.applyEdit(edit);
                
                // Open the file
                const doc = await vscode.workspace.openTextDocument(uri);
                await vscode.window.showTextDocument(doc);
                
                chatProvider.addMessage({
                    role: 'assistant',
                    content: `Created dbt model: \`${modelName}\`\n\n${model.documentation}`
                });
            } catch (error) {
                vscode.window.showErrorMessage(`ADAM Error: ${error}`);
            }
        })
    );

    // Voice chat
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.voice', async () => {
            const voiceEnabled = vscode.workspace.getConfiguration('adam').get('enableVoice');
            if (!voiceEnabled) {
                vscode.window.showInformationMessage('Voice is disabled. Enable it in settings.');
                return;
            }

            try {
                const voiceChat = new VoiceChat(adamClient as any);
                await voiceChat.start();
            } catch (error) {
                vscode.window.showErrorMessage(`ADAM Voice Error: ${error}`);
            }
        })
    );

    // Analyze data pattern
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.analyzeData', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return;
            }

            const selection = editor.document.getText(editor.selection) || editor.document.getText();
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is analyzing data patterns...",
                cancellable: false
            }, async (progress) => {
                try {
                    const analysis = await adamClient.analyzeDataPattern(selection);
                    
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: `## Data Analysis Results\n\n${analysis.summary}\n\n### Patterns Found:\n${analysis.patterns}\n\n### Recommendations:\n${analysis.recommendations}`
                    });
                    chatProvider.show();
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );
    
    // Select workspace folder
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.selectWorkspace', async () => {
            const result = await vscode.window.showOpenDialog({
                canSelectFiles: false,
                canSelectFolders: true,
                canSelectMany: false,
                openLabel: 'Select ADAM Backend Folder',
                defaultUri: vscode.Uri.file('/Users/vitoryago/ADAM')
            });
            
            if (result && result[0]) {
                const folderPath = result[0].fsPath;
                
                // Verify it's a valid ADAM backend folder
                if (!fs.existsSync(path.join(folderPath, 'src', 'adam_v2', 'main.py'))) {
                    vscode.window.showErrorMessage('Selected folder does not contain ADAM backend (src/adam_v2/main.py not found)');
                    return;
                }
                
                // Update configuration
                await vscode.workspace.getConfiguration('adam').update('backendPath', folderPath, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(`ADAM backend path updated to: ${folderPath}`);
                
                // Ask if user wants to restart backend
                const restart = await vscode.window.showInformationMessage(
                    'Restart ADAM backend with new path?',
                    'Yes', 'No'
                );
                
                if (restart === 'Yes') {
                    vscode.commands.executeCommand('adam.restartBackend');
                }
            }
        })
    );
    
    // Toggle standalone mode
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.toggleStandalone', async () => {
            const config = vscode.workspace.getConfiguration('adam');
            const currentMode = config.get('standalone', true);
            const newMode = !currentMode;
            
            await config.update('standalone', newMode, vscode.ConfigurationTarget.Global);
            
            const modeText = newMode ? 'Standalone (no backend)' : 'Backend mode';
            vscode.window.showInformationMessage(`ADAM switched to: ${modeText}`);
            
            // Reload window to apply changes
            const reload = await vscode.window.showInformationMessage(
                'Reload window to apply changes?',
                'Reload', 'Later'
            );
            
            if (reload === 'Reload') {
                vscode.commands.executeCommand('workbench.action.reloadWindow');
            }
        })
    );
    
    // Restart backend
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.restartBackend', async () => {
            const config = vscode.workspace.getConfiguration('adam');
            const useStandalone = config.get('standalone', true);
            
            if (useStandalone) {
                vscode.window.showInformationMessage('ADAM is in standalone mode - no backend to restart');
                return;
            }
            
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Restarting ADAM backend...",
                cancellable: false
            }, async () => {
                // Stop existing backend
                if (backendProcess) {
                    backendProcess.kill();
                    backendProcess = null;
                    await new Promise(resolve => setTimeout(resolve, 2000));  // Wait for process to die
                }
                
                // Start backend again
                const started = await startBackend(context);
                if (started) {
                    vscode.window.showInformationMessage('ADAM backend restarted successfully');
                } else {
                    vscode.window.showErrorMessage('Failed to restart ADAM backend');
                }
            });
        })
    );
}

export function deactivate() {
    if (adamClient) {
        adamClient.disconnect();
    }
    
    // Stop backend process if running
    if (backendProcess) {
        console.log('Stopping ADAM backend...');
        backendProcess.kill();
        backendProcess = null;
    }
}

/**
 * Start the ADAM backend automatically
 */
async function startBackend(context: vscode.ExtensionContext): Promise<boolean> {
    try {
        const config = vscode.workspace.getConfiguration('adam');
        const autoStart = config.get('autoStartBackend', true);
        
        if (!autoStart) {
            console.log('Auto-start backend is disabled');
            return false;
        }
        
        // Get backend path from configuration
        const configuredPath = config.get<string>('backendPath');
        
        // Try to find ADAM project path
        const adamPaths = [
            configuredPath,
            '/Users/vitoryago/ADAM',  // Primary ADAM location
            path.join(process.env.HOME || '', 'ADAM'),
            path.join(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', 'ADAM')
        ].filter(p => p);  // Remove undefined/null values
        
        let adamPath: string | null = null;
        for (const testPath of adamPaths) {
            if (testPath && fs.existsSync(path.join(testPath, 'src', 'adam_v2', 'main.py'))) {
                adamPath = testPath;
                break;
            }
        }
        
        if (!adamPath) {
            console.error('ADAM backend not found in expected locations');
            return false;
        }
        
        console.log(`Starting ADAM backend from: ${adamPath}`);
        
        // Check if backend is already running
        const isRunning = await checkBackendRunning();
        if (isRunning) {
            console.log('ADAM backend is already running');
            return true;
        }
        
        // Start the backend process
        backendProcess = spawn('python', ['-m', 'adam_v2.main'], {
            cwd: adamPath,
            env: {
                ...process.env,
                PYTHONPATH: path.join(adamPath, 'src')
            },
            detached: false
        });
        
        backendProcess.stdout?.on('data', (data) => {
            console.log(`ADAM Backend: ${data}`);
        });
        
        backendProcess.stderr?.on('data', (data) => {
            console.error(`ADAM Backend Error: ${data}`);
        });
        
        backendProcess.on('error', (error) => {
            console.error('Failed to start ADAM backend:', error);
            vscode.window.showErrorMessage(`Failed to start ADAM backend: ${error.message}`);
        });
        
        // Wait for backend to be ready
        await waitForBackend();
        
        vscode.window.showInformationMessage('ADAM backend started successfully');
        return true;
        
    } catch (error: any) {
        console.error('Error starting ADAM backend:', error);
        return false;
    }
}

/**
 * Check if backend is already running
 */
async function checkBackendRunning(): Promise<boolean> {
    try {
        const response = await fetch('http://localhost:8000/api/health');
        return response.ok;
    } catch {
        return false;
    }
}

/**
 * Wait for backend to be ready
 */
async function waitForBackend(maxRetries = 30): Promise<boolean> {
    for (let i = 0; i < maxRetries; i++) {
        const isRunning = await checkBackendRunning();
        if (isRunning) {
            return true;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    return false;
}