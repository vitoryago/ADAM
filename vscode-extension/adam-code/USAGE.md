# ADAM VSCode Extension - Usage Guide

## Quick Start

ADAM can now run automatically in every VS Code session, just like Claude Code!

### Standalone Mode (Default - No Backend Required)
By default, ADAM runs in standalone mode which doesn't require the backend. It will:
- Automatically load API keys from your `.env` file or VS Code settings
- Work immediately without any setup
- Use intelligent model routing (GPT-5/Grok)

### Backend Mode (Full Features)
For full memory persistence and advanced features, you can use backend mode:
1. Toggle to backend mode: Command Palette → "ADAM: Toggle Standalone Mode"
2. The backend will start automatically when VS Code opens
3. If needed, select custom backend folder: Command Palette → "ADAM: Select Workspace Folder"

## Key Features

### 1. Automatic Startup
- **Standalone Mode**: Works immediately when VS Code opens
- **Backend Mode**: Automatically starts the Python backend when VS Code opens

### 2. Workspace Selection
- Use Command Palette → "ADAM: Select Workspace Folder" to choose a different ADAM backend location
- Useful if you have multiple ADAM installations or custom setups

### 3. Mode Switching
- **Standalone Mode**: Fast, no backend needed, uses API keys directly
- **Backend Mode**: Full memory system, project isolation, advanced RAG
- Toggle with: Command Palette → "ADAM: Toggle Standalone Mode"

### 4. Backend Management
- **Auto-start**: Backend starts automatically (configurable in settings)
- **Manual restart**: Command Palette → "ADAM: Restart Backend"
- **Status**: Check the status bar for "ADAM Ready" indicator

## Configuration

Open VS Code Settings (Cmd+,) and search for "adam" to configure:

```json
{
  // Use standalone mode (no backend required)
  "adam.standalone": true,
  
  // Path to ADAM backend (for backend mode)
  "adam.backendPath": "/Users/vitoryago/ADAM",
  
  // Auto-start backend when extension activates
  "adam.autoStartBackend": true,
  
  // API Keys (for standalone mode)
  "adam.openaiApiKey": "your-key-here",
  "adam.grokApiKey": "your-key-here",
  
  // Preferred high-performance model
  "adam.preferredModel": "grok-4-reasoning"
}
```

## Commands

Access via Command Palette (Cmd+Shift+P):

- **ADAM: Open Chat** (Cmd+Shift+A) - Open ADAM chat panel
- **ADAM: Explain Code** (Cmd+Shift+E) - Explain selected code
- **ADAM: Toggle Standalone Mode** - Switch between standalone/backend modes
- **ADAM: Select Workspace Folder** - Choose ADAM backend location
- **ADAM: Restart Backend** - Restart the backend server
- **ADAM: Optimize SQL Query** - Optimize SQL in current file
- **ADAM: Generate dbt Model** - Create dbt models
- **ADAM: Voice Chat** - Start voice interaction

## Troubleshooting

### Extension Not Working?
1. Check if you have API keys configured (for standalone mode)
2. For backend mode, ensure Python and dependencies are installed
3. Check the Output panel (View → Output → ADAM) for errors

### Backend Won't Start?
1. Verify Python is installed: `python --version`
2. Check backend path in settings
3. Ensure dependencies are installed: `pip install -r requirements.txt`
4. Try manual start: `python -m adam_v2.main`

### Want to Use Different Folder?
1. Command Palette → "ADAM: Select Workspace Folder"
2. Navigate to your ADAM installation
3. Select the folder containing `src/adam_v2/main.py`

## Development

To develop or debug the extension:
1. Open `/Users/vitoryago/ADAM/vscode-extension/adam-code` in VS Code
2. Press F5 to launch Extension Development Host
3. Test your changes in the new VS Code window

## API Keys

### Standalone Mode
Add to VS Code settings or create `.env` file in ADAM root:
```env
OPENAI_API_KEY=your-openai-key
GROK_API_KEY=your-grok-key
```

### Backend Mode
The backend uses its own `.env` file in the ADAM root directory.