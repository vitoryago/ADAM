# ADAM Response Styles in VSCode

The VSCode extension now supports three response styles to customize how ADAM responds to your queries:

## Available Styles

### 📝 Normal (Default)
Balanced responses with moderate detail. Good for most use cases.

### ⚡ Concise
Brief, to-the-point responses. Perfect when you need quick answers without extra explanation.

### 📚 Explanatory
Detailed responses with thorough explanations. Ideal for learning and understanding complex concepts.

## How to Change Response Style

### Method 1: Using the Chat Interface
- Click the dropdown selector in the chat header
- Choose between Normal, Concise, or Explanatory
- The setting is saved automatically

### Method 2: Using Command Palette
- Press `Cmd+Shift+S` (or `Ctrl+Shift+S` on Windows/Linux)
- Or open Command Palette (`Cmd+Shift+P`) and search for "ADAM: Set Response Style"
- Select your preferred style from the options

### Method 3: In Settings
- Go to VSCode Settings
- Search for "adam.responseStyle"
- Choose your default style

## Examples

**Question:** "What does this function do?"

**Normal Response:**
"This function processes user input and validates it against a schema. It returns a validated object if successful, or throws an error if validation fails."

**Concise Response:**
"Validates user input against schema. Returns validated object or throws error."

**Explanatory Response:**
"This function serves as a validation layer for user input. It takes the raw input data and compares it against a predefined schema to ensure data integrity. The validation process includes:
1. Type checking for each field
2. Required field verification
3. Format validation for special fields (emails, dates, etc.)
4. Custom business rule validation

If all validations pass, it returns a cleaned and validated object. If any validation fails, it throws a descriptive error with details about what went wrong, helping developers debug issues quickly."

## Tips

- Use **Concise** for quick fixes and simple questions
- Use **Normal** for general coding assistance
- Use **Explanatory** when learning new concepts or debugging complex issues

The response style is persistent across sessions and applies to all ADAM interactions in VSCode.