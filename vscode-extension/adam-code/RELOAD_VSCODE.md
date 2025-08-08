# 🔄 Reload VSCode to Activate ADAM

## The extension has been updated and reinstalled!

### Now you need to:

1. **Reload VSCode Window**
   - Press `Cmd+Shift+P` to open Command Palette
   - Type "Reload Window"
   - Press Enter

   OR just close and reopen VSCode

2. **After reload, test ADAM:**
   - Press `Cmd+Shift+A` to open ADAM chat
   - Or press `Cmd+Shift+P` and type "ADAM" to see all commands

3. **Set your API key (if not done already):**
   - Press `Cmd+,` to open Settings
   - Search for "adam"
   - Add your OpenAI or Grok API key

## Troubleshooting

If commands still don't work after reload:

1. **Check extension is active:**
   - Press `Cmd+Shift+P`
   - Type "Show Running Extensions"
   - Look for "ADAM Code" in the list

2. **Check Output panel:**
   - View → Output
   - Select "ADAM Code" from dropdown
   - Look for any error messages

3. **Make sure you have an API key set:**
   - The extension needs either OpenAI or Grok API key to work
   - Set in Settings → search "adam"

## Version Info
- Version: 0.1.1
- Activation: onStartupFinished (activates after VSCode loads)
- Publisher: adamassistant

---

**Remember: You must reload VSCode after installing/updating the extension!**