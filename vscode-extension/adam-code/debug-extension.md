# How to Debug ADAM Extension

## Method 1: Using VS Code UI
1. Open VS Code
2. File → Open Folder → Select `/Users/vitoryago/ADAM/vscode-extension/adam-code`
3. Go to Run and Debug panel (Cmd+Shift+D)
4. At the top, you should see "Run Extension" in the dropdown
5. Click the green play button (or press F5)

## Method 2: Using Command Palette
1. Open the extension folder in VS Code
2. Press Cmd+Shift+P to open Command Palette
3. Type "Debug: Start Debugging"
4. Select "Run Extension"

## Method 3: Manual Launch
1. Open Terminal in VS Code
2. Run:
   ```bash
   code --extensionDevelopmentPath=/Users/vitoryago/ADAM/vscode-extension/adam-code
   ```

## If F5 doesn't work:
1. Make sure you have the extension folder open as the root folder in VS Code
2. Check that `.vscode/launch.json` exists
3. Try: View → Run → Start Debugging

## Alternative: Install Extension Locally
```bash
cd /Users/vitoryago/ADAM/vscode-extension/adam-code
code --install-extension .
```

## Check for issues:
1. View → Output → Select "Extension Host" from dropdown
2. Look for any error messages

## Testing without Debug Mode:
You can also package and install the extension:
```bash
npm install -g vsce
vsce package
code --install-extension adam-code-0.1.0.vsix
```