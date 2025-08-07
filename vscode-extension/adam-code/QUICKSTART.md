# ADAM Code - Quick Start Guide

## ✅ Installation Complete!

Your ADAM VS Code extension has been successfully compiled and is ready to use.

## 🚀 Launch the Extension

### Method 1: Using VS Code (Recommended)
1. Open VS Code
2. Open this folder: `/Users/vitoryago/ADAM/vscode-extension/adam-code`
3. Press `F5` to launch the Extension Development Host
4. A new VS Code window will open with ADAM loaded

### Method 2: Command Line
```bash
code /Users/vitoryago/ADAM/vscode-extension/adam-code
# Then press F5 in VS Code
```

## 🎯 First Steps

Once the extension is running:

1. **Open ADAM Chat**
   - Press `Cmd+Shift+A` (Mac) or `Ctrl+Shift+A` (Windows/Linux)
   - Or click the ADAM icon in the Activity Bar (left sidebar)

2. **Try Basic Commands**
   - Type "Hello ADAM" in the chat
   - Select some code and press `Cmd+Shift+E` to explain it
   - Open a SQL file and right-click → "ADAM: Optimize SQL Query"

## 🔧 Configuration

1. Open VS Code Settings (`Cmd+,`)
2. Search for "adam"
3. Configure:
   - `adam.serverUrl`: Your ADAM backend URL (default: http://localhost:8000)
   - `adam.projectId`: Your project ID
   - `adam.sqlDialect`: Your SQL dialect (bigquery, snowflake, etc.)

## ⚠️ Prerequisites

Make sure the ADAM backend is running:
```bash
cd /Users/vitoryago/ADAM/src/adam_v2
python main.py
```

## 🎤 Voice Setup (Optional)

To enable voice:
1. Set `adam.enableVoice` to `true` in settings
2. Ensure your OpenAI API key is configured in the backend
3. Use Command Palette → "ADAM: Voice Chat"

## 📝 Example Workflows

### SQL Optimization
```sql
-- 1. Open any .sql file
-- 2. Right-click in the editor
-- 3. Select "ADAM: Optimize SQL Query"
SELECT * FROM orders WHERE date > '2024-01-01'
```

### Generate dbt Model
1. Command Palette (`Cmd+Shift+P`)
2. Type "ADAM: Generate dbt Model"
3. Enter model name: `stg_customers`
4. Enter source table: `raw.customers`

### Create Feature Branch
1. Command Palette → "ADAM: Create Feature Branch"
2. Enter: "add customer metrics"
3. ADAM creates: `feature/add-customer-metrics`

## 🐛 Troubleshooting

### Extension not loading?
- Check the Output panel: View → Output → Select "ADAM" from dropdown
- Verify backend is running: `curl http://localhost:8000/api/health`

### Chat not responding?
- Check backend logs: Look for errors in the terminal running `main.py`
- Verify network: Can you access http://localhost:8000 in browser?

### Voice not working?
- Check `USE_OPENAI_TTS=true` in your .env file
- Verify OpenAI API key is set

## 🎉 Success!

You now have ADAM as your AI coworker directly in VS Code! 

Features ready to use:
- ✅ Intelligent chat with memory
- ✅ Code explanation
- ✅ SQL optimization
- ✅ dbt model generation
- ✅ Git workflow automation
- ✅ Voice interaction

Happy coding with ADAM! 🚀