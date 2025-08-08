# 🚀 ADAM VSCode Extension - Installation Guide

## ✅ Current Installation Status
- Extension installed: **adamassistant.adam-code v0.1.1**
- API Keys found in .env: **OpenAI ✅, xAI/Grok ✅**
- Standalone mode: **Enabled (no backend needed)**

## 🔄 IMPORTANT: Reload VSCode Now!

**You MUST reload VSCode for the extension to activate:**
1. Press `Cmd+Shift+P`
2. Type "Reload Window"
3. Press Enter

## 🎯 After Reload, Test ADAM

1. **Open ADAM Chat**: 
   - Press `Cmd+Shift+A`
   - Or click the ADAM icon in the left sidebar
   - Or Command Palette → "ADAM: Open Chat"

2. **Check Status Bar**: 
   - Look for "🧠 ADAM Ready" at bottom-right

3. **Test a Simple Message**:
   - Type "Hello" in the chat
   - ADAM should respond using your configured API keys

## Features in Standalone Mode

✅ **Works Without Backend** - No need to run any server or press F5!
✅ **File Reading** - ADAM can read and analyze your project files
✅ **Code Explanation** - Select code and get instant explanations
✅ **SQL Optimization** - Optimize queries with one click
✅ **Unified Memory** - Remembers context across sessions
✅ **Direct LLM Access** - Uses OpenAI or Grok directly

## Troubleshooting

### "No API keys configured"
Add your API key to VSCode settings:
1. Open settings: `Cmd+,`
2. Search for "adam"
3. Add either OpenAI or Grok API key

### Extension not activating
- Make sure you've run `npm run compile` before packaging
- Check the output panel (View > Output > ADAM Code) for errors

## Development Mode

If you want to modify and test the extension:
1. Open this folder in VSCode
2. Run `npm install`
3. Run `npm run compile`
4. Press `F5` to launch a test instance

## What's Different from Backend Mode?

Standalone mode (default):
- ✅ No backend server needed
- ✅ Works immediately after install
- ✅ Direct API calls to OpenAI/Grok
- ✅ Faster responses
- ✅ Works offline (for local file operations)

Backend mode (optional):
- Requires running ADAM backend server
- More complex routing between models
- Additional features like web search

## Memory Location

ADAM stores memories in: `~/.adam/memory/[project-name]/`
This ensures consistency across all interfaces (VSCode, Web, CLI)