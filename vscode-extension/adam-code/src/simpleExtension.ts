import * as vscode from 'vscode';
import { SimpleChatProvider } from './providers/simpleChatProvider';
import * as child_process from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let chatProvider: SimpleChatProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('ADAM Simple is activating...');
    
    // Auto-start backend if not running
    startBackendIfNeeded();
    
    // Register chat provider
    chatProvider = new SimpleChatProvider(context.extensionUri);
    
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            SimpleChatProvider.viewType,
            chatProvider
        )
    );
    
    // Register simple chat command
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.chat', async () => {
            await chatProvider.show();
        })
    );
    
    console.log('ADAM Simple is ready!');
}

function startBackendIfNeeded() {
    // Check if backend is running
    child_process.exec('curl -s http://localhost:8000/health', (error) => {
        if (error) {
            console.log('Backend not running, attempting to start...');
            
            const backendPath = '/Users/vitoryago/ADAM/src/adam_v2';
            if (fs.existsSync(backendPath)) {
                // Start backend in background
                const backend = child_process.spawn('python', ['main.py'], {
                    cwd: backendPath,
                    detached: true,
                    stdio: 'ignore'
                });
                
                backend.unref();
                
                vscode.window.showInformationMessage('Starting ADAM backend...');
                
                // Wait a bit for backend to start
                setTimeout(() => {
                    vscode.window.showInformationMessage('ADAM backend should be running now!');
                }, 5000);
            }
        } else {
            console.log('Backend is already running');
        }
    });
}

export function deactivate() {
    console.log('ADAM Simple deactivated');
}