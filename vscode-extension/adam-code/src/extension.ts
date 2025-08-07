import * as vscode from 'vscode';
import { ADAMChatProvider } from './providers/chatProvider';
import { ADAMClient } from './client/adamClient';
import { GitIntegration } from './integrations/gitIntegration';
import { SQLOptimizer } from './tools/sqlOptimizer';
import { DBTGenerator } from './tools/dbtGenerator';
import { FileManager } from './tools/fileManager';
import { VoiceChat } from './features/voiceChat';

let adamClient: ADAMClient;
let chatProvider: ADAMChatProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('ADAM Code is activating...');

    // Initialize ADAM client
    const config = vscode.workspace.getConfiguration('adam');
    // Use the actual project ID from your backend
    adamClient = new ADAMClient(
        config.get('serverUrl') || 'http://localhost:8000',
        config.get('projectId') || '3a859e97-16fd-46c6-b018-1ede9fade704'  // Your actual project ID
    );

    // Initialize chat provider
    chatProvider = new ADAMChatProvider(context.extensionUri, adamClient);

    // Register webview provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('adam.chatView', chatProvider)
    );

    // Register commands
    registerCommands(context);

    // Initialize features
    const gitIntegration = new GitIntegration();
    const sqlOptimizer = new SQLOptimizer(adamClient);
    const dbtGenerator = new DBTGenerator(adamClient);
    const fileManager = new FileManager();
    const voiceChat = new VoiceChat(adamClient);

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
}

function registerCommands(context: vscode.ExtensionContext) {
    // Main chat command
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.chat', () => {
            chatProvider.show();
        })
    );

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
                const voiceChat = new VoiceChat(adamClient);
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
}

export function deactivate() {
    if (adamClient) {
        adamClient.disconnect();
    }
}