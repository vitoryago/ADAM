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
    
    /**
     * List all stashes
     */
    async stashList(): Promise<{ success: boolean; stashes?: string[]; message?: string }> {
        const result = await this.execGit('stash list');
        
        if (result.success) {
            const stashes = result.output?.split('\n').filter(s => s.trim()) || [];
            return { success: true, stashes };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Apply specific stash without removing it
     */
    async stashApply(stashRef: string = 'stash@{0}'): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`stash apply ${stashRef}`);
        
        if (result.success) {
            return { success: true, message: `Applied stash: ${stashRef}` };
        }
        
        return { success: false, message: `Stash apply failed: ${result.error}` };
    }
    
    /**
     * Drop a specific stash
     */
    async stashDrop(stashRef: string = 'stash@{0}'): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`stash drop ${stashRef}`);
        
        if (result.success) {
            return { success: true, message: `Dropped stash: ${stashRef}` };
        }
        
        return { success: false, message: `Stash drop failed: ${result.error}` };
    }
    
    /**
     * Cherry-pick a commit
     */
    async cherryPick(commitHash: string, noCommit: boolean = false): Promise<{ success: boolean; message: string }> {
        const flags = noCommit ? '--no-commit' : '';
        const result = await this.execGit(`cherry-pick ${flags} ${commitHash}`);
        
        if (result.success) {
            return { success: true, message: `Cherry-picked commit: ${commitHash}` };
        }
        
        return { success: false, message: `Cherry-pick failed: ${result.error}` };
    }
    
    /**
     * Rebase current branch onto another
     */
    async rebase(targetBranch: string): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`rebase ${targetBranch}`);
        
        if (result.success) {
            return { success: true, message: `Rebased onto ${targetBranch}` };
        }
        
        return { success: false, message: `Rebase failed: ${result.error}` };
    }
    
    /**
     * Continue rebase after resolving conflicts
     */
    async rebaseContinue(): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit('rebase --continue');
        
        if (result.success) {
            return { success: true, message: 'Rebase continued' };
        }
        
        return { success: false, message: `Rebase continue failed: ${result.error}` };
    }
    
    /**
     * Abort ongoing rebase
     */
    async rebaseAbort(): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit('rebase --abort');
        
        if (result.success) {
            return { success: true, message: 'Rebase aborted' };
        }
        
        return { success: false, message: `Rebase abort failed: ${result.error}` };
    }
    
    /**
     * Reset to a specific commit
     */
    async reset(commitHash: string, mode: 'soft' | 'mixed' | 'hard' = 'mixed'): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`reset --${mode} ${commitHash}`);
        
        if (result.success) {
            return { success: true, message: `Reset to ${commitHash} (${mode})` };
        }
        
        return { success: false, message: `Reset failed: ${result.error}` };
    }
    
    /**
     * Revert a commit
     */
    async revert(commitHash: string): Promise<{ success: boolean; message: string }> {
        const result = await this.execGit(`revert ${commitHash} --no-edit`);
        
        if (result.success) {
            return { success: true, message: `Reverted commit: ${commitHash}` };
        }
        
        return { success: false, message: `Revert failed: ${result.error}` };
    }
    
    /**
     * Create a tag
     */
    async tag(tagName: string, message?: string, commitHash?: string): Promise<{ success: boolean; message: string }> {
        let command = message 
            ? `tag -a ${tagName} -m "${message}"`
            : `tag ${tagName}`;
            
        if (commitHash) {
            command += ` ${commitHash}`;
        }
        
        const result = await this.execGit(command);
        
        if (result.success) {
            return { success: true, message: `Created tag: ${tagName}` };
        }
        
        return { success: false, message: `Tag creation failed: ${result.error}` };
    }
    
    /**
     * List all tags
     */
    async listTags(): Promise<{ success: boolean; tags?: string[]; message?: string }> {
        const result = await this.execGit('tag');
        
        if (result.success) {
            const tags = result.output?.split('\n').filter(t => t.trim()) || [];
            return { success: true, tags };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Delete a tag
     */
    async deleteTag(tagName: string, remote: boolean = false): Promise<{ success: boolean; message: string }> {
        // Delete local tag
        const result = await this.execGit(`tag -d ${tagName}`);
        
        if (!result.success) {
            return { success: false, message: `Tag deletion failed: ${result.error}` };
        }
        
        // Delete remote tag if requested
        if (remote) {
            const remoteResult = await this.execGit(`push origin :refs/tags/${tagName}`);
            if (!remoteResult.success) {
                return { success: false, message: `Remote tag deletion failed: ${remoteResult.error}` };
            }
        }
        
        return { success: true, message: `Deleted tag: ${tagName}${remote ? ' (including remote)' : ''}` };
    }
    
    /**
     * Show diff of changes
     */
    async diff(staged: boolean = false): Promise<{ success: boolean; diff?: string; message?: string }> {
        const command = staged ? 'diff --cached' : 'diff';
        const result = await this.execGit(command);
        
        if (result.success) {
            return { success: true, diff: result.output || 'No changes' };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Get merge conflicts
     */
    async getConflicts(): Promise<{ success: boolean; conflicts?: string[]; message?: string }> {
        const result = await this.execGit('diff --name-only --diff-filter=U');
        
        if (result.success) {
            const conflicts = result.output?.split('\n').filter(f => f.trim()) || [];
            return { success: true, conflicts };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Fetch from remote
     */
    async fetch(all: boolean = false): Promise<{ success: boolean; message: string }> {
        const command = all ? 'fetch --all' : 'fetch';
        const result = await this.execGit(command);
        
        if (result.success) {
            return { success: true, message: 'Fetched from remote' };
        }
        
        return { success: false, message: `Fetch failed: ${result.error}` };
    }
    
    /**
     * Get remote branches
     */
    async getRemoteBranches(): Promise<{ success: boolean; branches?: string[]; message?: string }> {
        const result = await this.execGit('branch -r');
        
        if (result.success) {
            const branches = result.output?.split('\n')
                .filter(b => b.trim())
                .map(b => b.trim()) || [];
            return { success: true, branches };
        }
        
        return { success: false, message: result.error };
    }
    
    /**
     * Amend last commit
     */
    async amendCommit(newMessage?: string): Promise<{ success: boolean; message: string }> {
        const command = newMessage 
            ? `commit --amend -m "${newMessage}"`
            : 'commit --amend --no-edit';
            
        const result = await this.execGit(command);
        
        if (result.success) {
            return { success: true, message: 'Amended last commit' };
        }
        
        return { success: false, message: `Amend failed: ${result.error}` };
    }
    
    /**
     * Get workspace root path
     */
    private getWorkspaceRoot(): string {
        return this.workspacePath;
    }
}