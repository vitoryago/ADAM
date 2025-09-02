# ADAM VSCode Extension Troubleshooting

## Common Issues and Solutions

### "Command 'adam.setResponseStyle' not found"
**Solution:** 
1. Reload VSCode window (`Cmd+R` in VSCode or `code -r` in terminal)
2. Check if extension is active: Look for ADAM icon in Activity Bar
3. If not visible, reinstall: `code --install-extension adam-code-0.3.3.vsix --force`

### File context not working properly
**Problem:** ADAM doesn't understand which file you're referring to
**Solution:** The extension now automatically detects when you mention:
- "this file" / "this code"
- "shared" / "sharing"
- "show dependencies"
- "explain" / "analyze" / "review"

Make sure you have the file open in the active editor tab.

### Backend connection issues
**Symptoms:** 404 errors, health check failures
**Solution:**
1. Ensure backend is running: `cd src/adam_v2 && python main.py`
2. Check it's on port 8000: `curl http://localhost:8000/health`
3. Should return: `{"status":"ok"}`

### Extension not loading after install
**Solution:**
1. Remove old versions:
```bash
code --list-extensions | grep adam
code --uninstall-extension [old-extension-id]
```
2. Install fresh:
```bash
code --install-extension adam-code-0.3.3.vsix --force
code -r  # Reload VSCode
```

### Response style not changing
**Solution:**
1. Use the dropdown in chat header
2. Or press `Cmd+Shift+S` for command palette
3. Check setting is saved: Settings > Search "adam.responseStyle"

## Quick Checks

Run these commands to verify setup:

```bash
# Check extension is installed
code --list-extensions | grep adam

# Check backend is running
curl http://localhost:8000/health

# Check API is accessible
curl http://localhost:8000/api/health
```

## Complete Reinstall

If nothing works:
```bash
# 1. Uninstall all ADAM extensions
code --list-extensions | grep adam | xargs -L 1 code --uninstall-extension

# 2. Rebuild extension
cd /Users/vitoryago/ADAM/vscode-extension/adam-code
npm run compile
npx @vscode/vsce package --no-dependencies

# 3. Install fresh
code --install-extension adam-code-0.3.3.vsix --force

# 4. Reload VSCode
code -r
```

## Verify Everything Works

1. Press `Cmd+Shift+A` - ADAM chat should open
2. Press `Cmd+Shift+S` - Response style selector should appear
3. Type a message - Should get response from backend
4. Open a file and say "explain this" - Should include file content