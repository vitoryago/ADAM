import * as vscode from 'vscode';
import { ADAMClient } from '../client/adamClient';

export class VoiceChat {
    private isRecording: boolean = false;
    private statusBarItem: vscode.StatusBarItem;

    constructor(private adamClient: ADAMClient) {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            99
        );
    }

    async start(): Promise<void> {
        if (this.isRecording) {
            await this.stop();
        } else {
            await this.startRecording();
        }
    }

    private async startRecording(): Promise<void> {
        this.isRecording = true;
        this.statusBarItem.text = '$(unmute) Recording...';
        this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
        this.statusBarItem.show();

        vscode.window.showInformationMessage('🎤 ADAM is listening... Press again to stop.');

        // In a real implementation, this would:
        // 1. Start recording audio from microphone
        // 2. Stream to backend for transcription
        // 3. Get response and play TTS
        
        // For now, we'll use input box as fallback
        const input = await vscode.window.showInputBox({
            prompt: 'Speak to ADAM (type for now)',
            placeHolder: 'What would you like help with?'
        });

        if (input) {
            const response = await this.adamClient.sendMessage(input);
            
            // Show response
            vscode.window.showInformationMessage(`ADAM: ${response.content.substring(0, 100)}...`);
            
            // In real implementation, would play TTS here
        }

        await this.stop();
    }

    private async stop(): Promise<void> {
        this.isRecording = false;
        this.statusBarItem.hide();
        vscode.window.showInformationMessage('Recording stopped.');
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }
}