# Realistic BigQuery Problem-Solving Demos

These demos show ADAM solving real production issues that data engineers face daily.

## 🎬 Available Scenarios

### 1. **Dashboard Timeout Crisis** (`real_problem_demo.py`)
**Scenario**: CEO's dashboard failing at 3:47 AM, presentation in 2 hours
**Problem**: Query timeout after 180 seconds
**Solution**: ADAM identifies missing partition filter, reduces runtime to 4.2 seconds
**Best for**: Showing emergency response capabilities

```bash
python scripts/real_problem_demo.py
```

### 2. **Black Friday Pipeline Failure** (`pipeline_crisis_demo.py`) 
**Scenario**: Revenue pipeline crashes when order volume spikes 10x
**Problem**: "Resources Exceeded" errors blocking $2.3M daily reporting
**Solution**: ADAM implements micro-batching strategy
**Best for**: Demonstrating scale handling

```bash
python scripts/pipeline_crisis_demo.py
```

### 3. **Cost Explosion Investigation** (`cost_explosion_demo.py`)
**Scenario**: BigQuery costs jump from $1,000 to $10,000 per day
**Problem**: Someone removed a date filter, scanning 5 years of data
**Solution**: ADAM finds culprit query, saves $4.7M annually
**Best for**: ROI and cost control story

```bash
python scripts/cost_explosion_demo.py
```

### 4. **ADAM vs Manual Debugging** (`adam_vs_manual_demo.py`)
**Scenario**: Side-by-side comparison of traditional vs ADAM approach
**Shows**: 138 minutes manual work vs 45 seconds with ADAM
**Result**: 85% optimization vs 40% manual attempt
**Best for**: Demonstrating time savings

```bash
python scripts/adam_vs_manual_demo.py
```

### 5. **Preventing Disasters** (`proactive_prevention_demo.py`)
**Scenario**: Developer about to deploy a CROSS JOIN disaster
**Problem**: Query would cost $102,000/day and block production
**Solution**: ADAM catches it before deployment
**Best for**: Showing proactive protection

```bash
python scripts/proactive_prevention_demo.py
```

## 📹 Recording Tips

### For Maximum Impact:

1. **Start with the crisis** - Hook viewers immediately
2. **Show real numbers** - Costs, runtime, data volumes
3. **Keep it moving** - Each demo has built-in pacing
4. **Focus on outcomes** - Time saved, money saved, crisis averted

### Terminal Setup:
```bash
# Clear terminal
clear

# Make text larger (18-20pt recommended)
# Use dark terminal theme
# Hide all toolbars/tabs

# Run chosen demo
python scripts/cost_explosion_demo.py
```

## 📝 LinkedIn Post Templates

### For Dashboard Crisis:
```
At 3:47 AM, the CEO's dashboard crashed. Presentation at 6 AM.

Watch how ADAM diagnosed and fixed it in 8 minutes:
• Found the 185-second timeout query
• Identified missing partition filter  
• Reduced runtime to 4.2 seconds
• Dashboard back online ✓

This is why AI with memory matters.

#DataEngineering #BigQuery #AI #CrisisManagement
```

### For Cost Explosion:
```
$10,000 BigQuery bill in ONE DAY? 😱

ADAM found the problem in 30 seconds:
• Someone removed a date filter
• Query scanning 5 years instead of 30 days
• Running every 5 minutes

Fixed: $12,960/day → $12.48/day
Annual savings: $4.7 MILLION

#CloudCosts #BigQuery #DataEngineering #ROI
```

### For Prevention:
```
This query almost cost us $102,000 per day.

ADAM caught it before production:
❌ CROSS JOIN creating 100 trillion rows
❌ Would run for 6+ hours
❌ Block all other queries

✅ Corrected to 15 seconds, $1.20

AI that prevents disasters > AI that just responds

#BigQuery #DataEngineering #ProactiveTech
```

## 🎯 Key Messages

1. **ADAM solves REAL problems** - Not toy examples
2. **Speed matters** - Minutes vs hours of debugging  
3. **Money saved** - Concrete ROI numbers
4. **Learning system** - Gets better with each incident
5. **Proactive protection** - Prevents issues before they happen

## 🚀 Quick Start

Test the demos first:
```bash
cd docs/linkedin

# Most dramatic (30-40 seconds)
python scripts/cost_explosion_demo.py

# Fastest resolution (25-30 seconds)  
python scripts/real_problem_demo.py

# Best comparison (35-40 seconds)
python scripts/adam_vs_manual_demo.py
```

Choose based on your audience:
- **Technical**: Pipeline crisis or prevention demo
- **Business**: Cost explosion or ROI comparison
- **General**: Dashboard crisis (everyone understands "CEO presentation")

Remember: These aren't hypothetical - they're based on real BigQuery issues that cost companies millions.