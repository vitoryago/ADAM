# How to Test the Updated Extension

## Option 1: Run in Development Mode (Recommended for Testing)
1. Open VSCode
2. Open the extension folder: `/Users/vitoryago/ADAM/vscode-extension/adam-code`
3. Press `F5` to launch Extension Development Host
4. A new VSCode window will open with the development version
5. Press `Cmd+Shift+A` to open ADAM chat
6. Test with your crypto folder request

## Option 2: Package and Install Updated Extension
```bash
cd /Users/vitoryago/ADAM/vscode-extension/adam-code
npm run package  # Creates adam-code-0.3.1.vsix
code --install-extension adam-code-0.3.1.vsix --force
```
Then reload VSCode.

## Current Issue
The installed extension (0.3.1) doesn't have the timeout fix. You need to either:
- Run in development mode (F5)
- Or package and reinstall with the fixes

## Backend Issue
The backend is also hitting recursion limits. The orchestrator is working but needs optimization to handle the crypto folder more efficiently.