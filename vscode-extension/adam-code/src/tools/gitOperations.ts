/**
 * Git Operations for ADAM VSCode Extension
 * Provides git commit, push, branch capabilities
 */

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class GitOperations {
    private workspacePath: string;
    
    constructor() {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        this.workspacePath = workspaceFolder?.uri.fsPath || '';
    }
    
    /**
     * Execute a git command
     */
    private async execGit(command: string): Promise<{ success: boolean; output?: string; error?: string }> {
        try {
            const { stdout, stderr } = await execAsync(`git ${command}`, {
                cwd: this.workspacePath
            });
            
            if (stderr && !stderr.includes('warning')) {
                return { success: false, error: stderr };
            }
            
            return { success: true, output: stdout };
        } catch (error: any) {
            return { success: false, error: error.message };
        }
    }
    
    /**
     * Get current git status
     */
    async status(): Promise<{ success: boolean; status?: string; message?: string }> {
        const result = await this.execGit('status --short');
        
        if (result.success) {
            return { success: true, status: result.output || 'Working tree clean' };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Stage files for commit
     */
    async add(files: string[] = ['.']): Promise<{ success: boolean; message: string }> {
        const fileList = files.join(' ');
        const result = await this.execGit(`add ${fileList}`);
        
        if (result.success) {
            return { success: true, message: `Staged files: ${fileList}` };
        }
        
        return { success: false, message: `Failed to stage files: ${result.error}` };
    }
    
    /**
     * Create a commit
     */
    async commit(message: string, addAll: boolean = false): Promise<{ success: boolean; message: string }> {
        try {
            // Add all files if requested
            if (addAll) {
                await this.add(['.']);
            }
            
            // Check if there are staged changes
            const status = await this.execGit('diff --cached --name-only');
            if (!status.output || status.output.trim() === '') {
                return { success: false, message: 'No staged changes to commit' };
            }
            
            // Create commit with ADAM signature
            const commitMessage = `${message}\n\n🤖 Committed by ADAM VSCode Extension`;
            const result = await this.execGit(`commit -m "${commitMessage.replace(/"/g, '\\"')}"`);
            
            if (result.success) {
                // Get commit hash
                const hashResult = await this.execGit('rev-parse HEAD');
                const hash = hashResult.output?.trim().substring(0, 7);
                return { success: true, message: `Created commit ${hash}: ${message}` };
            }
            
            return { success: false, message: `Commit failed: ${result.error}` };
        } catch (error: any) {
            return { success: false, message: `Commit error: ${error.message}` };
        }
    }
    
    /**
     * Push to remote
     */
    async push(force: boolean = false): Promise<{ success: boolean; message: string }> {
        const forceFlag = force ? '--force-with-lease' : '';
        const result = await this.execGit(`push ${forceFlag}`);
        
        if (result.success) {
            return { success: true, message: 'Pushed to remote successfully' };
        }
        
        // Check if we need to set upstream
        if (result.error?.includes('no upstream')) {
            const branch = await this.getCurrentBranch();
            const upstreamResult = await this.execGit(`push --set-upstream origin ${branch}`);
            
            if (upstreamResult.success) {
                return { success: true, message: `Pushed to remote and set upstream for ${branch}` };
            }
        }
        
        return { success: false, message: `Push failed: ${result.error}` };
    }
    
    /**
     * Pull from remote
     */
    async pull(): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit('pull');
        
        if (result.success) {
            return { success: true, message: 'Pulled from remote successfully' };
        }
        
        return { success: false, message: `Pull failed: ${result.error}` };
    }
    
    /**
     * Create and checkout a new branch
     */
    async createBranch(branchName: string): Promise<{ success: boolean; message: string }> {
        // Sanitize branch name
        const sanitized = branchName
            .toLowerCase()
            .replace(/\s+/g, '-')
            .replace(/[^a-z0-9-_\/]/g, '');
        
        const result = await this.execGit(`checkout -b ${sanitized}`);
        
        if (result.success) {
            return { success: true, message: `Created and switched to branch: ${sanitized}` };
        }
        
        // Check if branch already exists
        if (result.error?.includes('already exists')) {
            const checkoutResult = await this.execGit(`checkout ${sanitized}`);
            if (checkoutResult.success) {
                return { success: true, message: `Switched to existing branch: ${sanitized}` };
            }
        }
        
        return { success: false, message: `Failed to create branch: ${result.error}` };
    }
    
    /**
     * Switch to a different branch
     */
    async checkout(branchName: string): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`checkout ${branchName}`);
        
        if (result.success) {
            return { success: true, message: `Switched to branch: ${branchName}` };
        }
        
        return { success: false, message: `Failed to switch branch: ${result.error}` };
    }
    
    /**
     * Get current branch name
     */
    async getCurrentBranch(): Promise<string> {
        const result = await this.execGit('branch --show-current');
        return result.output?.trim() || 'main';
    }
    
    /**
     * List all branches
     */
    async listBranches(): Promise<{ success: boolean; branches?: string[]; message?: string }> {
        const result = await this.execGit('branch -a');
        
        if (result.success) {
            const branches = result.output?.split('\n')
                .filter(b => b.trim())
                .map(b => b.replace(/^\*?\s+/, '')) || [];
            return { success: true, branches };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Get recent commits
     */
    async log(limit: number = 10): Promise<{ success: boolean; commits?: string[]; message?: string }> {
        const result = await this.execGit(`log --oneline -${limit}`);
        
        if (result.success) {
            const commits = result.output?.split('\n').filter(c => c.trim()) || [];
            return { success: true, commits };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Stash changes
     */
    async stash(message?: string): Promise<{ success: boolean; message: string }> {
        const stashMessage = message ? `push -m "${message}"` : 'push';
        const result = await this.execGit(`stash ${stashMessage}`);
        
        if (result.success) {
            return { success: true, message: 'Changes stashed' };
        }
        
        return { success: false, message: `Stash failed: ${result.error}` };
    }
    
    /**
     * Apply stashed changes
     */
    async stashPop(): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit('stash pop');
        
        if (result.success) {
            return { success: true, message: 'Stashed changes applied' };
        }
        
        return { success: false, message: `Stash pop failed: ${result.error}` };
    }
}