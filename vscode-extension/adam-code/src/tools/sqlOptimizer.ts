import * as vscode from 'vscode';
import { ADAMClient } from '../client/adamClient';

export class SQLOptimizer {
    constructor(private adamClient: ADAMClient) {}

    async optimizeCurrentFile() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active SQL file');
            return;
        }

        if (editor.document.languageId !== 'sql') {
            vscode.window.showWarningMessage('Current file is not SQL');
            return;
        }

        const sql = editor.document.getText();
        const dialect = await this.detectDialect(sql);

        try {
            const optimization = await this.adamClient.optimizeSQL(sql, dialect);
            
            // Show optimization in new document
            const doc = await vscode.workspace.openTextDocument({
                language: 'sql',
                content: optimization.query
            });
            
            await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            
            vscode.window.showInformationMessage(
                `SQL optimized! ${optimization.estimatedSpeedup}`
            );
        } catch (error) {
            vscode.window.showErrorMessage(`Optimization failed: ${error}`);
        }
    }

    private async detectDialect(sql: string): Promise<string> {
        // Simple dialect detection
        const sqlLower = sql.toLowerCase();
        
        if (sqlLower.includes('unnest') || sqlLower.includes('array[')) {
            return 'bigquery';
        } else if (sqlLower.includes('variant') || sqlLower.includes('flatten')) {
            return 'snowflake';
        } else if (sqlLower.includes('redshift')) {
            return 'redshift';
        } else if (sqlLower.includes('pg_') || sqlLower.includes('postgres')) {
            return 'postgresql';
        } else {
            return 'standard';
        }
    }
}