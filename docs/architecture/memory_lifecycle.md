# ADAM Memory Lifecycle System

## Overview

ADAM's memory lifecycle system implements intelligent memory management with decay, reinforcement, and compression strategies inspired by human memory patterns. This system ensures ADAM's memory remains efficient and relevant over time.

## Key Concepts

### 1. Memory Strength

Each memory has a "strength" value that:
- Starts at 1.0 when created
- Decays exponentially over **active days** (0.95^active_days)
- Gets reinforced when accessed (+0.1 boost)
- Never falls below 0.01

**Important**: Memory decay is based on "active days" not calendar days. If you don't use ADAM for a week, your memories won't decay during that time. Only days when you interact with ADAM count towards memory aging.

### 2. Memory Tiers

Memories are classified into tiers based on strength and **active age**:

- **Active** (0-7 active days): Full fidelity, quick access
- **Archive** (7-30 active days): Still accessible but lower priority
- **Compress** (30+ active days): Eligible for compression
  - `compress_moderate`: Remove redundancy, keep substance
  - `compress_high`: Extract only key insights
  - `compress_ultra`: Single paragraph summary
- **Landmark**: Never compressed, marked as permanently important

Note: These are "active days" - days when you actually used ADAM, not calendar days.

### 3. Reinforcement

When memories are recalled:
- They receive a strength boost proportional to relevance
- More relevant memories get stronger reinforcement
- Reinforcement prevents decay and compression

## Usage

### Automatic Reinforcement

Every time ADAM recalls a memory, it's automatically reinforced:

```python
# In recall_with_context method
boost = 0.1 * similarity  # More relevant = stronger boost
new_strength = lifecycle_manager.reinforce_memory(memory_id, metadata, boost)
```

### Manual Decay Cycle

Apply decay to all memories:

```bash
./scripts/manage_memory_lifecycle.py decay
```

### Check Memory Health

View memory system statistics:

```bash
./scripts/manage_memory_lifecycle.py health
```

### View Activity Report

See your usage patterns and active days:

```bash
./scripts/manage_memory_lifecycle.py activity
```

### Mark Landmark Memories

Prevent important memories from being compressed:

```bash
./scripts/manage_memory_lifecycle.py landmark <memory_id>
```

## Memory Importance Calculation

Importance is calculated based on multiple factors:

- **Strength** (30%): Current decayed strength value
- **Access Frequency** (20%): How often the memory is accessed
- **Success Rate** (20%): How well solutions worked
- **Has Code** (15%): Code patterns are more valuable
- **Reference Count** (10%): How often referenced by other memories
- **User Marked** (5%): Explicitly marked as landmark

## Active Days System

ADAM tracks "active days" to ensure memories only age when the system is being used:

- **Active Day**: Any day where you have at least one interaction with ADAM
- **Memory Age**: Calculated as number of active days since memory creation
- **Vacation Safe**: If you don't use ADAM for weeks, memories won't decay
- **Fair Aging**: All memories age at the same rate relative to actual usage

This prevents the situation where you return from vacation to find all your memories compressed.

## Compression Strategy

When memories are compressed (based on active days):

### 1. **Moderate Compression** (7-30 active days)
- **Reduction**: ~20-30% size reduction
- **Preserves**: All questions, answers, code snippets, errors, solutions
- **Removes**: Pleasantries, redundant explanations, verbose descriptions
- **Example**: Full SQL query examples kept, but lengthy explanations shortened

### 2. **High Compression** (30-90 active days)
- **Reduction**: ~50-60% size reduction  
- **Format**: "Problem: X | Solution: Y | Key insight: Z"
- **Preserves**: Core problem, final solution, critical code (max 3 lines)
- **Removes**: Detailed explanations, intermediate steps, context

### 3. **Ultra Compression** (90+ active days)
- **Reduction**: ~90-95% size reduction
- **Format**: Single sentence, maximum 20 words
- **Preserves**: What was learned or solved
- **Removes**: All details, code, explanations

### LLM-Based Compression

ADAM uses grok-3-mini to intelligently compress memories:

```python
# The system analyzes content and preserves based on type
compressor = MemoryCompressor()
result = await compressor.compress_memory(content, metadata, level)

# Result includes:
# - compressed_content: The compressed text
# - compression_ratio: How much was saved (0.0-1.0)
# - preserved_elements: What was kept
# - tokens_saved: Estimated token reduction
```

### Compression Commands

Manually compress eligible memories:
```bash
# Compress memories that need it
./scripts/manage_memory_lifecycle.py compress

# Force recompression of all non-landmark memories
./scripts/manage_memory_lifecycle.py compress --force
```

### Compression Validation

The system validates each compression to ensure:
- Essential information is preserved
- Compression actually reduces size
- Format matches the compression level
- Searchability is maintained

## Benefits

1. **Efficient Storage**: 80-90% reduction for old memories
2. **Better Recall**: Recent and important memories prioritized
3. **Natural Forgetting**: Mimics human memory patterns
4. **Cost Effective**: Reduces embedding and storage costs
5. **Self-Improving**: Frequently used memories strengthen

## Configuration

Set environment variables to customize behavior:

```bash
# Decay rate (default: 0.95)
export ADAM_DECAY_RATE=0.95

# Archive threshold (default: 0.3)
export ADAM_ARCHIVE_THRESHOLD=0.3

# Compression threshold (default: 0.1)
export ADAM_COMPRESS_THRESHOLD=0.1
```

## Future Enhancements

1. **LLM-Based Compression**: Use language models for intelligent summarization
2. **Automatic Scheduling**: Background decay cycles
3. **Memory Consolidation**: Merge similar memories
4. **Importance Learning**: Learn what makes memories valuable
5. **User Preferences**: Personalized decay rates

## Example Workflow

1. User asks a question
2. ADAM searches memories (with decay applied)
3. Relevant memories are reinforced
4. New valuable responses are stored
5. Periodic decay cycles clean up old memories
6. Important memories survive through reinforcement

This creates a self-organizing memory system that keeps relevant information accessible while efficiently managing storage.