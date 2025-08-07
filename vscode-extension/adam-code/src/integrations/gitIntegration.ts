import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class GitIntegration {
    async getCurrentBranch(): Promise<string> {
        try {
            const { stdout } = await execAsync('git branch --show-current');
            return stdout.trim();
        } catch (error) {
            throw new Error(`Failed to get current branch: ${error}`);
        }
    }

    async createBranch(branchName: string): Promise<void> {
        try {
            await execAsync(`git checkout -b ${branchName}`);
        } catch (error) {
            throw new Error(`Failed to create branch: ${error}`);
        }
    }

    async getChanges(): Promise<string> {
        try {
            const { stdout } = await execAsync('git diff HEAD');
            return stdout;
        } catch (error) {
            throw new Error(`Failed to get changes: ${error}`);
        }
    }

    async commit(message: string): Promise<void> {
        try {
            await execAsync(`git add .`);
            await execAsync(`git commit -m "${message}"`);
        } catch (error) {
            throw new Error(`Failed to commit: ${error}`);
        }
    }
}