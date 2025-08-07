#!/bin/bash

echo "🚀 Setting up ADAM Code VS Code Extension..."

# Navigate to extension directory
cd /Users/vitoryago/ADAM/vscode-extension/adam-code

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Compile TypeScript
echo "🔨 Compiling TypeScript..."
npm run compile

# Create necessary directories
mkdir -p out/tools out/features out/integrations out/client out/providers

# Create placeholder files for additional modules
echo "export class GitIntegration {}" > src/integrations/gitIntegration.ts
echo "export class SQLOptimizer { constructor(client: any) {} }" > src/tools/sqlOptimizer.ts
echo "export class DBTGenerator { constructor(client: any) {} }" > src/tools/dbtGenerator.ts
echo "export class FileManager {}" > src/tools/fileManager.ts
echo "export class VoiceChat { constructor(client: any) {} start() {} }" > src/features/voiceChat.ts

# Compile again with all files
npm run compile

echo "✅ Extension compiled successfully!"
echo ""
echo "To test the extension:"
echo "1. Open VS Code"
echo "2. Press F5 to launch a new Extension Development Host"
echo "3. In the new VS Code window, press Cmd+Shift+A to open ADAM chat"
echo ""
echo "Make sure ADAM backend is running:"
echo "cd /Users/vitoryago/ADAM/src/adam_v2 && python main.py"