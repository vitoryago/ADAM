# 🔧 ADAM Extension Troubleshooting Guide

## ✅ Current Status
- **Extension installed:** adamassistant.adam-code v0.1.1
- **Installation successful:** The extension is properly installed in VSCode

## 🚨 Issue: "Command 'adam.chat' not found"

This happens when VSCode hasn't activated the extension yet. Here's how to fix it:

### Solution 1: Reload VSCode Window (Recommended)
1. **With VSCode open**, press `Cmd+Shift+P`
2. Type "Reload Window" and press Enter
3. Wait for VSCode to restart
4. Try `Cmd+Shift+A` again

### Solution 2: Full VSCode Restart
1. **Close ALL VSCode windows** completely (Cmd+Q)
2. Open Terminal and run:
   ```bash
   "/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code" .
   ```
3. Once VSCode opens, try `Cmd+Shift+A`

### Solution 3: Check Extension Activation
1. Open VSCode
2. Go to View → Output
3. In the dropdown, select "Extension Host"
4. Look for "ADAM Code is activating..." message
5. If you see errors, note them down

## 🔍 Verify Extension is Active

### Check Commands Are Registered:
1. Press `Cmd+Shift+P` to open Command Palette
2. Type "ADAM" - you should see:
   - ADAM: Open Chat
   - ADAM: Explain Code
   - ADAM: Optimize SQL Query
   - ADAM: Create Feature Branch
   - ADAM: Create Pull Request

### Check Status Bar:
- Look at the bottom-right of VSCode
- You should see "🧠 ADAM Ready" in the status bar
- Click it to open chat

### Check Activity Bar:
- On the left sidebar, look for the ADAM icon
- Click it to open the ADAM panel

## 🔑 Configure API Key

The extension won't work without an API key:

1. Press `Cmd+,` to open Settings
2. Search for "adam"
3. Add ONE of these:
   - `adam.openaiApiKey`: Your OpenAI API key
   - `adam.grokApiKey`: Your Grok/xAI API key
4. Save settings (Cmd+S)
5. Reload window (Cmd+Shift+P → "Reload Window")

## 📊 Check Extension Logs

If still having issues:

1. Open VSCode
2. Press `Cmd+Shift+P`
3. Type "Developer: Show Logs..."
4. Select "Extension Host"
5. Look for lines containing "ADAM"

You should see:
```
ADAM Code is activating...
ADAM running in standalone mode - no backend required!
ADAM Code is ready!
```

## 🆘 Still Not Working?

### Manual Command Test:
```bash
# Check extension is installed
"/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code" --list-extensions | grep adam

# Should output: adamassistant.adam-code
```

### Force Reinstall:
```bash
# Uninstall
"/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code" --uninstall-extension adamassistant.adam-code

# Reinstall
"/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code" --install-extension /Users/vitoryago/ADAM/vscode-extension/adam-code/adam-code-0.1.1.vsix

# Open VSCode
"/Users/vitoryago/Downloads/Visual Studio Code.app/Contents/Resources/app/bin/code" .
```

## ✨ When It's Working

You'll know ADAM is working when:
1. `Cmd+Shift+A` opens the ADAM chat panel
2. "ADAM Ready" appears in the status bar
3. ADAM commands appear in Command Palette
4. The ADAM icon appears in the activity bar

## 🎯 Quick Test

Once ADAM is working, try this:
1. Open any code file
2. Select some code
3. Press `Cmd+Shift+E` to explain the selected code
4. ADAM should analyze and explain the code

---

**Remember:** The most common fix is simply reloading the VSCode window!