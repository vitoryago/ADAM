import * as vscode from 'vscode';
import { ADAMChatProvider } from './providers/chatProvider';
import { ADAMClient } from './client/adamClient';
import { GitIntegration } from './integrations/gitIntegration';
import { SQLOptimizer } from './tools/sqlOptimizer';
import { DBTGenerator } from './tools/dbtGenerator';
import { FileManager } from './tools/fileManager';
import { VoiceChat } from './features/voiceChat';
import { DeepDiscussionProvider } from './providers/deepDiscussionProvider';

let adamClient: ADAMClient;
let chatProvider: ADAMChatProvider;
let deepDiscussionProvider: DeepDiscussionProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('ADAM Code is activating...');

    // Initialize ADAM client
    const config = vscode.workspace.getConfiguration('adam');
    // Use the actual project ID from your backend
    adamClient = new ADAMClient(
        config.get('serverUrl') || 'http://localhost:8000',
        config.get('projectId') || '788d358e-4089-405b-9614-237583ea5dc2'  // VSCode Extension project
    );

    // Set initial response style from settings
    const responseStyle = config.get<'normal' | 'concise' | 'explanatory'>('responseStyle') || 'normal';
    adamClient.setResponseStyle(responseStyle);

    // Initialize chat provider
    chatProvider = new ADAMChatProvider(context.extensionUri, adamClient);

    // Register webview provider
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('adam.chatView', chatProvider)
    );

    deepDiscussionProvider = new DeepDiscussionProvider(context.extensionUri, adamClient);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('adam.deepDiscussionView', deepDiscussionProvider)
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

    // Response style command
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.setResponseStyle', async () => {
            const style = await vscode.window.showQuickPick(
                [
                    { label: '📝 Normal', description: 'Balanced responses with moderate detail', value: 'normal' },
                    { label: '⚡ Concise', description: 'Brief, to-the-point responses', value: 'concise' },
                    { label: '📚 Explanatory', description: 'Detailed responses with thorough explanations', value: 'explanatory' }
                ],
                {
                    placeHolder: 'Select ADAM response style',
                    title: 'ADAM Response Style'
                }
            );

            if (style) {
                adamClient.setResponseStyle(style.value as 'normal' | 'concise' | 'explanatory');
                vscode.window.showInformationMessage(`ADAM response style set to: ${style.label}`);
                
                // Update the setting
                const config = vscode.workspace.getConfiguration('adam');
                await config.update('responseStyle', style.value, true);
            }
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

    // DBT Documentation Commands

    // Document current model columns
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.dbtDocumentModel', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !editor.document.fileName.endsWith('.sql')) {
                vscode.window.showInformationMessage('Please open a DBT model file (.sql)');
                return;
            }

            const modelName = editor.document.fileName.split('/').pop()?.replace('.sql', '');
            if (!modelName) return;

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `ADAM is documenting columns for ${modelName}...`,
                cancellable: false
            }, async (progress) => {
                try {
                    const result = await adamClient.documentDBTColumns(modelName);

                    // Show documentation in chat
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: `## Column Documentation for ${modelName}\n\n${result.documentation}\n\n### Next Steps:\n- Review the generated descriptions\n- Save to schema.yml when satisfied\n- Run \`dbt docs generate\` to update documentation`
                    });
                    chatProvider.show();

                    // Ask if user wants to save to schema.yml
                    const save = await vscode.window.showInformationMessage(
                        `Generated documentation for ${result.columns_documented} columns. Save to schema.yml?`,
                        'Yes', 'No'
                    );

                    if (save === 'Yes') {
                        await adamClient.saveDBTDocumentation(modelName, result.yaml_content);
                        vscode.window.showInformationMessage('Documentation saved to schema.yml');
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    // Analyze column patterns across project
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.dbtAnalyzeColumns', async () => {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is analyzing column patterns across your DBT project...",
                cancellable: false
            }, async (progress) => {
                try {
                    const analysis = await adamClient.analyzeDBTColumns();

                    // Show analysis in chat
                    chatProvider.addMessage({
                        role: 'assistant',
                        content: `## DBT Column Analysis Report\n\n### Summary\n- Total Columns: ${analysis.summary.total_columns}\n- Documented: ${analysis.summary.documented_columns} (${analysis.summary.documentation_coverage})\n- Common Columns: ${analysis.summary.common_columns}\n\n### Top Common Columns\n${analysis.common_columns.map((c: any) => `- **${c.name}**: Appears in ${c.occurrence_count} models${c.has_documentation ? ' ✓' : ' ⚠️ Undocumented'}`).join('\n')}\n\n### Detected Patterns\n${Object.entries(analysis.patterns_detected).map(([pattern, cols]: [string, any]) => `- **${pattern}**: ${cols.length} columns`).join('\n')}\n\n### Recommendations\n- Focus on documenting the ${analysis.summary.undocumented_common_columns} undocumented common columns first\n- Use consistent descriptions for columns that appear in multiple models\n- Add relationship tests for detected foreign keys`
                    });
                    chatProvider.show();
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    // Find and standardize common columns
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.dbtStandardizeColumns', async () => {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is finding columns that need standardization...",
                cancellable: false
            }, async (progress) => {
                try {
                    const commonColumns = await adamClient.findCommonDBTColumns();

                    if (commonColumns.length === 0) {
                        vscode.window.showInformationMessage('No common columns found that need standardization');
                        return;
                    }

                    // Show quick pick to select column to standardize
                    const selected = await vscode.window.showQuickPick(
                        commonColumns.map((c: any) => ({
                            label: c.column_name,
                            description: `Appears in ${c.occurrence_count} models`,
                            detail: c.needs_standardization ? '⚠️ Has inconsistent descriptions' : '✓ Consistent',
                            column: c
                        })),
                        {
                            placeHolder: 'Select a column to standardize documentation'
                        }
                    );

                    if (selected) {
                        const standardDesc = await adamClient.standardizeColumnDescription(
                            selected.column.column_name,
                            selected.column.suggested_standard_description
                        );

                        vscode.window.showInformationMessage(
                            `Standardized documentation for '${selected.column.column_name}' across ${selected.column.occurrence_count} models`
                        );
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`ADAM Error: ${error}`);
                }
            });
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('adam.deepDiscussion', () => {
            vscode.commands.executeCommand('adam.deepDiscussionView.focus');
        })
    );

    // Generate schema.yml for current directory
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.dbtGenerateSchema', async () => {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }

            // Get the current folder path
            const activeFile = vscode.window.activeTextEditor?.document.uri;
            const folderPath = activeFile ? vscode.Uri.joinPath(activeFile, '..').fsPath : workspaceFolder.uri.fsPath;

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is generating schema.yml with intelligent column descriptions...",
                cancellable: false
            }, async (progress) => {
                try {
                    const yamlContent = await adamClient.generateDBTSchema(folderPath);

                    // Create schema.yml file
                    const schemaPath = vscode.Uri.joinPath(vscode.Uri.file(folderPath), 'schema.yml');
                    const edit = new vscode.WorkspaceEdit();

                    // Check if file exists
                    try {
                        await vscode.workspace.fs.stat(schemaPath);
                        // File exists, ask to overwrite
                        const overwrite = await vscode.window.showWarningMessage(
                            'schema.yml already exists. Overwrite?',
                            'Yes', 'No', 'Merge'
                        );

                        if (overwrite === 'No') {
                            return;
                        } else if (overwrite === 'Merge') {
                            // TODO: Implement merge logic
                            vscode.window.showInformationMessage('Merge feature coming soon!');
                            return;
                        }
                    } catch {
                        // File doesn't exist, create it
                    }

                    edit.createFile(schemaPath, {
                        contents: Buffer.from(yamlContent),
                        overwrite: true
                    });
                    await vscode.workspace.applyEdit(edit);

                    // Open the file
                    const doc = await vscode.workspace.openTextDocument(schemaPath);
                    await vscode.window.showTextDocument(doc);

                    vscode.window.showInformationMessage('Generated schema.yml with intelligent column descriptions');
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