/**
 * Tool Integration for ADAM VSCode Extension
 * Enables ADAM to execute file operations and code generation directly from VSCode
 */

import * as vscode from 'vscode';
import { ADAMClient } from '../client/adamClient';

export interface ToolRequest {
    tool: string;
    params: Record<string, any>;
}

export interface ToolResponse {
    status: 'success' | 'error' | 'warning';
    data: any;
    message: string;
    metadata?: Record<string, any>;
}

export class ADAMToolIntegration {
    constructor(private adamClient: ADAMClient) {}

    /**
     * Process natural language request and execute tools
     */
    async processToolRequest(request: string): Promise<ToolResponse> {
        // Send request to ADAM backend for tool planning
        const response = await this.adamClient.sendMessage(
            `TOOL_REQUEST: ${request}`,
            true
        );

        // Parse tool execution plan from response
        const toolPlan = this.parseToolPlan(response.content);
        
        if (toolPlan) {
            return await this.executeTool(toolPlan);
        }

        return {
            status: 'error',
            data: null,
            message: 'Could not determine tool to use'
        };
    }

    /**
     * Execute a specific tool
     */
    async executeTool(toolRequest: ToolRequest): Promise<ToolResponse> {
        const { tool, params } = toolRequest;

        switch (tool) {
            case 'read_file':
                return await this.readFile(params.file_path);
            
            case 'write_file':
                return await this.writeFile(params.file_path, params.content);
            
            case 'edit_file':
                return await this.editFile(
                    params.file_path,
                    params.old_text,
                    params.new_text
                );
            
            case 'generate_code':
                return await this.generateCode(params);
            
            case 'create_dag':
                return await this.createDAG(params);
            
            case 'optimize_sql':
                return await this.optimizeSQL(params.query, params.dialect);
            
            default:
                // Send to backend for execution
                return await this.executeRemoteTool(toolRequest);
        }
    }

    /**
     * Read file contents
     */
    private async readFile(filePath: string): Promise<ToolResponse> {
        try {
            const uri = vscode.Uri.file(filePath);
            const content = await vscode.workspace.fs.readFile(uri);
            const text = Buffer.from(content).toString('utf8');
            
            // Add line numbers
            const lines = text.split('\n');
            const numberedContent = lines.map((line, i) => 
                `${(i + 1).toString().padStart(4, ' ')}→${line}`
            ).join('\n');

            return {
                status: 'success',
                data: numberedContent,
                message: `Read ${lines.length} lines from ${filePath}`,
                metadata: {
                    line_count: lines.length,
                    file_path: filePath
                }
            };
        } catch (error) {
            return {
                status: 'error',
                data: null,
                message: `Failed to read file: ${error}`
            };
        }
    }

    /**
     * Write content to file
     */
    private async writeFile(filePath: string, content: string): Promise<ToolResponse> {
        try {
            const uri = vscode.Uri.file(filePath);
            const contentBuffer = Buffer.from(content, 'utf8');
            
            // Check if file exists
            const exists = await vscode.workspace.fs.stat(uri).then(
                () => true,
                () => false
            );

            // Write file
            await vscode.workspace.fs.writeFile(uri, contentBuffer);

            // Open the file in editor
            const doc = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(doc);

            return {
                status: 'success',
                data: content,
                message: `${exists ? 'Updated' : 'Created'} ${filePath}`,
                metadata: {
                    action: exists ? 'update' : 'create',
                    file_path: filePath,
                    line_count: content.split('\n').length
                }
            };
        } catch (error) {
            return {
                status: 'error',
                data: null,
                message: `Failed to write file: ${error}`
            };
        }
    }

    /**
     * Edit file by replacing text
     */
    private async editFile(
        filePath: string,
        oldText: string,
        newText: string
    ): Promise<ToolResponse> {
        try {
            const uri = vscode.Uri.file(filePath);
            const doc = await vscode.workspace.openTextDocument(uri);
            const editor = await vscode.window.showTextDocument(doc);
            
            const fullText = doc.getText();
            const startIndex = fullText.indexOf(oldText);
            
            if (startIndex === -1) {
                return {
                    status: 'error',
                    data: null,
                    message: 'Text to replace not found in file'
                };
            }

            // Create edit
            const edit = new vscode.WorkspaceEdit();
            const startPos = doc.positionAt(startIndex);
            const endPos = doc.positionAt(startIndex + oldText.length);
            const range = new vscode.Range(startPos, endPos);
            
            edit.replace(uri, range, newText);
            
            // Apply edit
            const success = await vscode.workspace.applyEdit(edit);
            
            if (success) {
                await doc.save();
                return {
                    status: 'success',
                    data: newText,
                    message: `Replaced text in ${filePath}`,
                    metadata: {
                        file_path: filePath,
                        replaced: oldText,
                        with: newText
                    }
                };
            } else {
                return {
                    status: 'error',
                    data: null,
                    message: 'Failed to apply edit'
                };
            }
        } catch (error) {
            return {
                status: 'error',
                data: null,
                message: `Failed to edit file: ${error}`
            };
        }
    }

    /**
     * Generate code using ADAM
     */
    private async generateCode(params: any): Promise<ToolResponse> {
        const prompt = `Generate ${params.language || 'Python'} code for: ${params.requirements}`;
        const response = await this.adamClient.sendMessage(prompt);
        
        // Extract code from response
        const codeMatch = response.content.match(/```[\w]*\n([\s\S]*?)```/);
        const code = codeMatch ? codeMatch[1] : response.content;
        
        // Save to file if specified
        if (params.output_file) {
            await this.writeFile(params.output_file, code);
        }
        
        return {
            status: 'success',
            data: code,
            message: `Generated ${params.language || 'Python'} code`,
            metadata: {
                language: params.language || 'Python',
                output_file: params.output_file
            }
        };
    }

    /**
     * Create Airflow DAG
     */
    private async createDAG(params: any): Promise<ToolResponse> {
        const prompt = `Create an Airflow DAG named '${params.dag_name}' with schedule '${params.schedule || '@daily'}' and these tasks: ${JSON.stringify(params.tasks)}`;
        const response = await this.adamClient.sendMessage(prompt);
        
        // Extract DAG code
        const codeMatch = response.content.match(/```python\n([\s\S]*?)```/);
        const dagCode = codeMatch ? codeMatch[1] : response.content;
        
        // Save to file if specified
        if (params.output_file) {
            await this.writeFile(params.output_file, dagCode);
        }
        
        return {
            status: 'success',
            data: dagCode,
            message: `Created DAG '${params.dag_name}'`,
            metadata: {
                dag_name: params.dag_name,
                task_count: params.tasks.length,
                output_file: params.output_file
            }
        };
    }

    /**
     * Optimize SQL query
     */
    private async optimizeSQL(query: string, dialect: string = 'postgresql'): Promise<ToolResponse> {
        const response = await this.adamClient.optimizeSQL(query, dialect);
        
        return {
            status: 'success',
            data: response.query,
            message: 'SQL query optimized',
            metadata: {
                original: query,
                optimized: response.query,
                explanation: response.explanation
            }
        };
    }

    /**
     * Execute tool on ADAM backend
     */
    private async executeRemoteTool(toolRequest: ToolRequest): Promise<ToolResponse> {
        try {
            const response = await this.adamClient.sendMessage(
                `EXECUTE_TOOL: ${JSON.stringify(toolRequest)}`,
                true
            );
            
            // Parse tool response from ADAM
            const toolResponse = this.parseToolResponse(response.content);
            return toolResponse || {
                status: 'success',
                data: response.content,
                message: 'Tool executed remotely'
            };
        } catch (error) {
            return {
                status: 'error',
                data: null,
                message: `Remote tool execution failed: ${error}`
            };
        }
    }

    /**
     * Parse tool plan from ADAM response
     */
    private parseToolPlan(content: string): ToolRequest | null {
        try {
            // Look for JSON in the response
            const jsonMatch = content.match(/\{[\s\S]*"tool"[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        } catch (error) {
            console.error('Failed to parse tool plan:', error);
        }
        return null;
    }

    /**
     * Parse tool response from ADAM
     */
    private parseToolResponse(content: string): ToolResponse | null {
        try {
            // Look for tool response format in content
            const jsonMatch = content.match(/\{[\s\S]*"status"[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        } catch (error) {
            console.error('Failed to parse tool response:', error);
        }
        return null;
    }
}

/**
 * Register tool commands in VSCode
 */
export function registerToolCommands(
    context: vscode.ExtensionContext,
    adamClient: ADAMClient
) {
    const toolIntegration = new ADAMToolIntegration(adamClient);

    // Register command for natural language tool requests
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.executeTools', async () => {
            const request = await vscode.window.showInputBox({
                prompt: 'What would you like ADAM to do?',
                placeHolder: 'e.g., "Create a Python script that processes CSV files"'
            });

            if (!request) {
                return;
            }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "ADAM is working on your request...",
                cancellable: false
            }, async () => {
                try {
                    const result = await toolIntegration.processToolRequest(request);
                    
                    if (result.status === 'success') {
                        vscode.window.showInformationMessage(
                            `ADAM: ${result.message}`
                        );
                    } else {
                        vscode.window.showErrorMessage(
                            `ADAM Error: ${result.message}`
                        );
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(
                        `ADAM Error: ${error}`
                    );
                }
            });
        })
    );

    // Register quick actions
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.readCurrentFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showInformationMessage('No file is currently open');
                return;
            }

            const result = await toolIntegration.executeTool({
                tool: 'read_file',
                params: { file_path: editor.document.fileName }
            });

            vscode.window.showInformationMessage(
                `ADAM read ${result.metadata?.line_count} lines from ${editor.document.fileName}`
            );
        })
    );

    return toolIntegration;
}