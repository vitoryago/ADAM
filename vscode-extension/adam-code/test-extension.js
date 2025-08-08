#!/usr/bin/env node

const { exec } = require('child_process');
const path = require('path');

console.log('🔍 Testing ADAM VSCode Extension...\n');

const codePath = '/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code';

// Check if extension is installed
exec(`"${codePath}" --list-extensions`, (error, stdout, stderr) => {
    if (error) {
        console.error('❌ Error checking extensions:', error);
        return;
    }
    
    const extensions = stdout.split('\n');
    const adamExtension = extensions.find(ext => ext.includes('adam'));
    
    if (adamExtension) {
        console.log('✅ ADAM extension is installed:', adamExtension);
    } else {
        console.log('❌ ADAM extension not found in installed extensions');
        console.log('Installed extensions:', extensions.filter(e => e).join(', '));
    }
});

console.log('\n📝 Instructions to test ADAM in VSCode:');
console.log('1. Close all VSCode windows');
console.log('2. Open VSCode fresh: code .');
console.log('3. Press Cmd+Shift+P to open Command Palette');
console.log('4. Type "ADAM" to see all ADAM commands');
console.log('5. Select "ADAM: Open Chat" or press Cmd+Shift+A');
console.log('\nIf commands don\'t appear:');
console.log('- Check View → Output → Select "Extension Host" from dropdown');
console.log('- Look for any ADAM-related errors');
console.log('- Try: Developer: Reload Window from Command Palette');