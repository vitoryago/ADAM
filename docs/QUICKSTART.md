# ADAM Quick Start Guide

## First Time Setup (10 minutes)

1. **Install Dependencies**
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

3. **Configure ADAM**
   ```bash
   cp config/.env.template .env
   # Edit .env with your preferences
   ```

4. **Meet ADAM**
   ```bash
   python src/hello_adam.py
   ```

## Daily Development Flow

1. **Start your day**
   ```bash
   source venv/bin/activate
   git pull  # Get latest changes
   ```

2. **Work on ADAM**
   - Experiment in `notebooks/`
   - Build features in `src/adam/`
   - Document learnings in `docs/daily_logs/`

3. **End your day**
   ```bash
   git add .
   git commit -m "Day X: What I accomplished"
   git push
   ```

## Testing Ideas

Ask ADAM:
- "How do I write a CTE?"
- "Explain window functions"
- "Help me optimize this query"
- "What's wrong with my JOIN?"

## Troubleshooting

- **No audio?** Check microphone permissions
- **Slow responses?** Try the smaller llama2 model
- **Import errors?** Ensure venv is activated
