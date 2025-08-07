# ADAM Code - Your AI Analytics Engineering Coworker

ADAM Code brings your intelligent AI assistant directly into VS Code, giving you a coworker that remembers everything and helps with your day-to-day analytics engineering tasks.

## Features

### 🧠 **Intelligent Chat Interface**
- Chat with ADAM directly in VS Code
- Full conversation history with memory
- Context-aware responses based on your current file and project

### 💻 **Code Understanding**
- **Explain Code** - Select any code and get detailed explanations
- **Code Review** - Get suggestions for improvements
- **Documentation Generation** - Auto-generate docs for your functions

### 🚀 **SQL Optimization**
- Automatically optimize SQL queries for your dialect (BigQuery, Snowflake, etc.)
- Get performance improvement suggestions
- See side-by-side comparisons of original vs optimized queries

### 📊 **dbt Integration**
- Generate dbt models from source tables
- Create staging models with best practices
- Auto-generate documentation and tests

### 🔀 **Git Workflow Automation**
- **Smart Branch Creation** - ADAM formats branch names following conventions
- **PR Generation** - Automatically create PRs with detailed descriptions
- **Commit Messages** - Generate meaningful commit messages

### 🎤 **Voice Interaction**
- Talk to ADAM using voice commands
- Get spoken responses
- Perfect for when you're thinking through problems

### 📈 **Data Analysis**
- Analyze data patterns in your files
- Get insights and recommendations
- Identify anomalies and trends

## Installation

1. Install the extension from VS Code Marketplace (coming soon)
2. Make sure ADAM backend is running:
   ```bash
   cd /Users/vitoryago/ADAM/src/adam_v2
   python main.py
   ```
3. Configure the extension in VS Code settings

## Usage

### Quick Start

1. **Open Chat**: Press `Cmd+Shift+A` (Mac) or `Ctrl+Shift+A` (Windows/Linux)
2. **Explain Code**: Select code and press `Cmd+Shift+E`
3. **Optimize SQL**: Right-click in any SQL file and select "ADAM: Optimize SQL Query"

### Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `ADAM: Open Chat` | `Cmd+Shift+A` | Open ADAM chat panel |
| `ADAM: Explain Code` | `Cmd+Shift+E` | Explain selected code |
| `ADAM: Optimize SQL Query` | - | Optimize current SQL file |
| `ADAM: Create Feature Branch` | - | Create a new git branch |
| `ADAM: Create Pull Request` | - | Generate PR with description |
| `ADAM: Generate dbt Model` | - | Create dbt staging model |
| `ADAM: Voice Chat` | - | Start voice conversation |

### Configuration

```json
{
  "adam.serverUrl": "http://localhost:8000",
  "adam.projectId": "analytics-project",
  "adam.enableVoice": true,
  "adam.autoMemory": true,
  "adam.sqlDialect": "bigquery",
  "adam.dbtProject": "/path/to/dbt/project"
}
```

## Analytics Engineering Workflows

### 1. SQL Development
```sql
-- Select your query and right-click → "ADAM: Optimize SQL Query"
SELECT * FROM customers WHERE status = 'active'
```

ADAM will:
- Analyze the query structure
- Suggest indexes
- Optimize JOIN orders
- Add proper partitioning

### 2. dbt Model Creation
```
Command: "ADAM: Generate dbt Model"
Input: Model name and source table
Output: Complete staging model with tests
```

### 3. Data Quality Checks
Select your data sample and use "ADAM: Analyze Data Pattern" to:
- Identify missing values
- Find anomalies
- Suggest data quality tests

### 4. Documentation
ADAM automatically:
- Documents your SQL queries
- Explains complex business logic
- Maintains a knowledge base of your data models

## Memory Network

ADAM remembers:
- All your conversations
- Code patterns you use
- Common issues and solutions
- Project-specific knowledge

This memory persists across sessions and grows stronger over time.

## Integration with Analytics Tools

ADAM integrates with:
- **BigQuery** - Query optimization, cost estimation
- **dbt** - Model generation, testing, documentation
- **Git** - Branch management, PR creation
- **Looker** - LookML generation (coming soon)
- **Airflow** - DAG generation (coming soon)

## Tips

1. **Use Project Context**: Set your project ID in settings for project-specific memory
2. **Voice for Brainstorming**: Use voice chat when thinking through complex problems
3. **Memory Search**: Ask ADAM about past solutions - it remembers everything
4. **SQL Dialect**: Set your SQL dialect for accurate optimizations

## Privacy & Security

- All conversations are stored locally in your ADAM instance
- No data is sent to external services (except configured LLMs)
- Memory is project-isolated for security

## Troubleshooting

### ADAM not responding?
1. Check the backend is running: `python /Users/vitoryago/ADAM/src/adam_v2/main.py`
2. Verify server URL in settings matches your backend
3. Check the Output panel (View → Output → ADAM)

### Voice not working?
1. Ensure `adam.enableVoice` is true in settings
2. Check microphone permissions
3. Verify OpenAI API key is set for TTS

## Coming Soon

- [ ] Inline code suggestions
- [ ] Looker LookML generation
- [ ] Airflow DAG creation
- [ ] Data lineage visualization
- [ ] Team knowledge sharing
- [ ] Cost optimization recommendations

## Support

For issues or questions:
- GitHub: [ADAM Repository](https://github.com/yourusername/ADAM)
- Documentation: [ADAM Docs](https://adam-docs.com)

---

Built with ❤️ for Analytics Engineers by Analytics Engineers