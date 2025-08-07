import * as vscode from 'vscode';
import { ADAMClient } from '../client/adamClient';

export class SQLOptimizer {
    constructor(private adamClient: ADAMClient) {}

    async optimizeCurrentFile(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'sql') {
            vscode.window.showWarningMessage('Please open a SQL file');
            return;
        }

        const query = editor.document.getText();
        const dialect = vscode.workspace.getConfiguration('adam').get('sqlDialect') as string;

        try {
            const optimized = await this.adamClient.optimizeSQL(query, dialect);
            
            // Show optimization in new editor
            const doc = await vscode.workspace.openTextDocument({
                content: optimized.query,
                language: 'sql'
            });
            
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            
            vscode.window.showInformationMessage('SQL optimized! Check the new tab for results.');
        } catch (error) {
            vscode.window.showErrorMessage(`Optimization failed: ${error}`);
        }
    }
}