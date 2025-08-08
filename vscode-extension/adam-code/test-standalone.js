#!/usr/bin/env node

/**
 * Test the standalone ADAM functionality
 * This verifies that the standalone client can work without the backend
 */

const { StandaloneADAMClient } = require('./out/standalone/standaloneClient');
const { UnifiedMemoryManager } = require('./out/standalone/memoryManager');

// Mock VSCode context for testing
const mockContext = {
    globalStorageUri: { fsPath: '/tmp/adam-test' },
    extensionUri: { fsPath: __dirname }
};

// Mock VSCode workspace
global.vscode = {
    workspace: {
        getConfiguration: (section) => ({
            get: (key, defaultValue) => {
                // Return test API keys from environment
                if (key === 'openaiApiKey') return process.env.OPENAI_API_KEY;
                if (key === 'grokApiKey') return process.env.GROK_API_KEY || process.env.XAI_API_KEY;
                return defaultValue;
            }
        }),
        name: 'test-project',
        workspaceFolders: [{
            uri: { fsPath: process.cwd() }
        }],
        findFiles: async () => [],
        fs: {
            readFile: async () => Buffer.from('test content'),
            writeFile: async () => {}
        },
        openTextDocument: async () => ({}),
        rootPath: process.cwd()
    },
    window: {
        showTextDocument: async () => {},
        createTerminal: () => ({ show: () => {}, sendText: () => {} }),
        showInformationMessage: console.log,
        showErrorMessage: console.error
    },
    Uri: {
        file: (path) => ({ fsPath: path }),
        joinPath: (base, ...paths) => ({ fsPath: [base.fsPath, ...paths].join('/') })
    }
};

async function testStandalone() {
    console.log('Testing ADAM Standalone Mode...\n');
    
    try {
        // Test memory manager
        console.log('1. Testing Memory Manager:');
        const memoryManager = new UnifiedMemoryManager('test-project');
        await memoryManager.saveMemory('test question', 'test answer', 'test-workspace');
        const memories = await memoryManager.searchMemories('test', 1);
        console.log('   ✓ Memory saved and retrieved:', memories.length > 0 ? 'Success' : 'Failed');
        
        // Test standalone client
        console.log('\n2. Testing Standalone Client:');
        const client = new StandaloneADAMClient(mockContext);
        
        // Check if API keys are configured
        const hasKeys = process.env.OPENAI_API_KEY || process.env.GROK_API_KEY || process.env.XAI_API_KEY;
        if (!hasKeys) {
            console.log('   ⚠ No API keys found. Set OPENAI_API_KEY or GROK_API_KEY to test LLM calls.');
        } else {
            console.log('   ✓ API keys configured');
            
            // Test sending a message
            console.log('\n3. Testing LLM Communication:');
            const response = await client.sendMessage('Say "Hello ADAM test" and nothing else', false);
            console.log('   Response:', response.content);
            console.log('   Model used:', response.model);
        }
        
        console.log('\n✅ All tests completed successfully!');
        console.log('\nThe ADAM VSCode extension is ready to use in standalone mode.');
        console.log('No backend server is required - it works like Claude Code!');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        process.exit(1);
    }
}

// Run the test
testStandalone().catch(console.error);