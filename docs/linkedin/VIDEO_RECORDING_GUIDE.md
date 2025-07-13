# Video Recording Guide for LinkedIn BigQuery Demos

## Quick Demo Scripts (Choose One)

### 1. **10-Second Demo** (`instant_demo.py`)
```bash
python scripts/instant_demo.py
```
- Shows before/after metrics only
- Perfect for LinkedIn feed autoplay
- Key message: 87% performance improvement

### 2. **25-Second Demo** (`realtime_analysis.py`)
```bash
python scripts/realtime_analysis.py
```
- Shows ADAM analyzing in real-time
- Progress indicators create visual interest
- Ends with concrete optimizations

### 3. **Video-Optimized Demo** (`video_demo.py`)
```bash
python scripts/video_demo.py
```
- Clear screen transitions
- Typing effect for engagement
- Best for 30-45 second videos

### 4. **SQL Transformation** (`sql_transformation_demo.py`)
```bash
python scripts/sql_transformation_demo.py
```
- Shows actual SQL before/after
- Good for technical audience
- Demonstrates concrete changes

### 5. **Batch Results** (`batch_optimization_demo.py`)
```bash
python scripts/batch_optimization_demo.py
```
- Multiple queries optimized
- Shows cumulative impact
- Great for "scale" messaging

## Recording Tips

### Terminal Setup
```bash
# Set clean terminal
clear
# Adjust font size (recommend 16-18pt)
# Use dark theme for better contrast
# Hide terminal tabs/toolbars if possible
```

### Recording Process
1. **Test Run**: Do a practice run first
2. **Clean Start**: Clear terminal before recording
3. **Steady Pace**: Don't rush through output
4. **Focus Area**: Zoom in on key metrics

### Best Practices for LinkedIn

#### For 10-second video:
- Start recording with terminal ready
- Run `instant_demo.py`
- Pause 2 seconds on final metrics
- No audio needed (use captions)

#### For 25-second video:
- Add quick intro: "Watch ADAM optimize this query"
- Run `realtime_analysis.py`
- Highlight the improvement percentages
- End with call-to-action

#### For 30-45 second video:
- Use `video_demo.py` for best visual flow
- Add voiceover if desired
- Include your LinkedIn handle as overlay

## Sample Video Scripts

### 10-Second Script (Text Overlay)
```
"BigQuery too slow?"
[Show metrics]
"ADAM finds optimizations in seconds"
[Show results]
"87% faster. Every time."
```

### 25-Second Script (Voiceover)
```
"This BigQuery query takes 2 minutes and costs $3.50.
Let's see how ADAM optimizes it.
[Run demo]
ADAM found 5 similar cases and identified three optimizations.
Result: 94% faster, 94% cheaper.
This is the power of AI with memory."
```

## Post-Recording

### Edit Suggestions
- Trim dead space at start/end
- Add LinkedIn handle watermark
- Include captions for accessibility
- Keep final video under 60 seconds

### LinkedIn Post Example
```
🚀 BigQuery running slow? Watch ADAM optimize it in real-time.

From 120 seconds → 7 seconds
From $3.50 → $0.20 per query

How? ADAM remembers every optimization and applies learned patterns to new queries.

#BigQuery #AI #DataEngineering #PerformanceOptimization
```

## Quick Command Reference

```bash
# Navigate to demo directory
cd docs/linkedin

# For quick metrics display
python scripts/instant_demo.py

# For technical audience
python scripts/sql_transformation_demo.py

# For visual engagement
python scripts/video_demo.py

# To show scale
python scripts/batch_optimization_demo.py
```

Remember: The goal is to show dramatic improvement quickly and clearly. Focus on the numbers - they tell the story!