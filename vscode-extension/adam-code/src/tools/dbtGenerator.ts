import * as vscode from 'vscode';
import * as path from 'path';
import { ADAMClient } from '../client/adamClient';

export class DBTGenerator {
    constructor(private adamClient: ADAMClient) {}

    async generateModel(modelName: string, sourceTable: string): Promise<void> {
        try {
            const model = await this.adamClient.generateDBTModel(modelName, sourceTable);
            
            // Get dbt project path
            const dbtProject = vscode.workspace.getConfiguration('adam').get('dbtProject') as string;
            if (!dbtProject) {
                vscode.window.showErrorMessage('Please configure dbt project path in settings');
                return;
            }

            // Create model file
            const modelPath = path.join(dbtProject, 'models', 'staging', `${modelName}.sql`);
            const uri = vscode.Uri.file(modelPath);
            
            const edit = new vscode.WorkspaceEdit();
            edit.createFile(uri, { contents: Buffer.from(model.sql) });
            await vscode.workspace.applyEdit(edit);
            
            // Open the file
            const doc = await vscode.workspace.openTextDocument(uri);
            await vscode.window.showTextDocument(doc);
            
            vscode.window.showInformationMessage(`dbt model ${modelName} created successfully!`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to generate dbt model: ${error}`);
        }
    }
}