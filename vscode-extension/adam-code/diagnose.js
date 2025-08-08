#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔍 ADAM VSCode Extension Diagnostic\n');
console.log('=' .repeat(50));

// 1. Check if extension is installed
console.log('\n1️⃣ Extension Installation:');
try {
    const codePath = '/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code';
    const extensions = execSync(`"${codePath}" --list-extensions 2>/dev/null`, { encoding: 'utf8' });
    const adamExt = extensions.split('\n').find(ext => ext.includes('adam'));
    if (adamExt) {
        console.log('   ✅ Extension installed:', adamExt);
    } else {
        console.log('   ❌ Extension NOT installed');
    }
} catch (e) {
    console.log('   ❌ Error checking extensions:', e.message);
}

// 2. Check compiled output
console.log('\n2️⃣ Compiled JavaScript:');
const outDir = path.join(__dirname, 'out');
if (fs.existsSync(outDir)) {
    const files = fs.readdirSync(outDir);
    console.log(`   ✅ Found ${files.length} compiled files`);
    console.log('   Files:', files.join(', '));
} else {
    console.log('   ❌ No compiled output found');
}

// 3. Check package.json
console.log('\n3️⃣ Package Configuration:');
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
console.log('   Version:', packageJson.version);
console.log('   Main:', packageJson.main);
console.log('   Activation:', packageJson.activationEvents?.join(', ') || 'None');

// 4. Check API keys in .env
console.log('\n4️⃣ API Keys (.env):');
const envPath = '/Users/vitoryago/ADAM/.env';
if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf8');
    const hasOpenAI = envContent.includes('OPENAI_API_KEY=');
    const hasGrok = envContent.includes('GROK_API_KEY=') || envContent.includes('XAI_API_KEY=');
    console.log('   OpenAI API Key:', hasOpenAI ? '✅ Found' : '❌ Not found');
    console.log('   Grok/xAI API Key:', hasGrok ? '✅ Found' : '❌ Not found');
} else {
    console.log('   ❌ .env file not found');
}

// 5. Check media files
console.log('\n5️⃣ WebView Resources:');
const mediaDir = path.join(__dirname, 'media');
if (fs.existsSync(mediaDir)) {
    const mediaFiles = fs.readdirSync(mediaDir);
    console.log(`   ✅ Found ${mediaFiles.length} media files`);
    mediaFiles.forEach(f => console.log(`      - ${f}`));
} else {
    console.log('   ❌ Media directory not found');
}

// 6. Check VSIX package
console.log('\n6️⃣ VSIX Package:');
const vsixFiles = fs.readdirSync(__dirname).filter(f => f.endsWith('.vsix'));
if (vsixFiles.length > 0) {
    console.log('   ✅ Found VSIX packages:');
    vsixFiles.forEach(f => {
        const stats = fs.statSync(path.join(__dirname, f));
        console.log(`      - ${f} (${(stats.size / 1024).toFixed(1)} KB)`);
    });
} else {
    console.log('   ❌ No VSIX packages found');
}

console.log('\n' + '=' .repeat(50));
console.log('\n📋 Diagnosis Summary:\n');

// Provide diagnosis
console.log('Issues found:');
console.log('1. The extension is in standalone mode but may not be properly initializing');
console.log('2. API keys are available in .env but may not be loading correctly');
console.log('3. The WebView might be failing to load the chat interface');

console.log('\n🔧 Recommended Fix:\n');
console.log('1. Rebuild the extension with proper error handling:');
console.log('   npm run compile');
console.log('   npx vsce package --no-dependencies');
console.log('\n2. Reinstall with verbose logging:');
console.log('   code --uninstall-extension adamassistant.adam-code');
console.log('   code --install-extension adam-code-0.1.1.vsix');
console.log('\n3. Reload VSCode and check Output panel:');
console.log('   View → Output → Select "ADAM Code" from dropdown');
console.log('\n4. Try the command directly:');
console.log('   Cmd+Shift+P → "ADAM: Open Chat"');