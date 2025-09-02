# Installing ADAM VSCode Extension

## Quick Install (Without F5)

### Method 1: Command Line Install
```bash
code --install-extension adam-code-0.3.2.vsix
```

### Method 2: VSCode UI Install
1. Open VSCode
2. Go to Extensions view (Cmd+Shift+X)
3. Click the "..." menu at the top of Extensions panel
4. Select "Install from VSIX..."
5. Navigate to `/Users/vitoryago/ADAM/vscode-extension/adam-code/`
6. Select `adam-code-0.3.2.vsix`
7. Click Install

### Method 3: Drag and Drop
Simply drag the `adam-code-0.3.2.vsix` file into your VSCode window

## After Installation

1. **Reload VSCode** - Click "Reload" when prompted or restart VSCode
2. **Open ADAM Chat** - Press `Cmd+Shift+A` or click the ADAM icon in the Activity Bar
3. **Set Response Style** - Press `Cmd+Shift+S` to choose your preferred response style

## Verify Installation

1. Look for the ADAM icon in the Activity Bar (left sidebar)
2. Check the status bar for "ADAM Ready" indicator
3. Open Command Palette and search for "ADAM" commands

## Troubleshooting

### If ADAM chat doesn't open:
- Make sure the backend is running: `cd src/adam_v2 && python main.py`
- Check that it's running on http://localhost:8000

### If file context isn't working:
- Make sure you have a file open in the editor
- The extension now automatically detects when you're referring to the active file

### To uninstall previous versions:
```bash
code --uninstall-extension adam-code
```

## Features Now Available

✅ **Response Styles** - Normal, Concise, or Explanatory modes
✅ **Better File Context** - Automatically includes active file when relevant
✅ **Persistent Settings** - Your preferences are saved
✅ **No F5 Required** - Install and use like any other extension

## Updating the Extension

When you make changes:
1. `npm run compile` - Compile TypeScript
2. `npx @vscode/vsce package` - Create new VSIX
3. Reinstall the new VSIX file