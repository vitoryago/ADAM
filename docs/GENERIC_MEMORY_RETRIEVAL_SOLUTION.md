# Generic Memory Retrieval Solution

## Problem Statement
When users ask generic questions about past work (e.g., "show me what we did", "bring back that thing"), ADAM often retrieves old, highly-reinforced memories instead of recent relevant ones.

## Root Cause
1. **Strength Imbalance**: Old memories accumulate high strength (0.95+) through repeated access
2. **Generic Matching**: Vague queries match many memories equally
3. **No Recency Bias**: Default system doesn't prioritize recent over old

## Generic Solution (Domain-Agnostic)

### 1. Intent Detection Enhancement
The system now detects generic recall patterns that work for ANY domain:

```python
# Generic patterns (not specific to DAGs or any particular content)
generic_recall_patterns = [
    "bring me back", "show me", "can you bring",
    "we have done", "we created", "we discussed", "we talked about",
    "any", "some", "that", "those"
]

# Specificity indicators that suggest user knows what they want
specificity_indicators = [
    "specific", "particular", "exact", "called", "named",
    "with", "contains", "includes", "about"
]
```

### 2. Smart Query Enhancement
Only enhance queries that are:
- Generic (match recall patterns)
- Lack temporal hints (no "recent", "last", etc.)
- Lack specificity (no specific names or details)

```python
if is_generic_recall and not has_temporal_hint and not has_specificity:
    enhanced_query = f"{prompt} (focusing on our most recent conversations)"
```

### 3. Recency-Based Scoring
For generic queries, recent memories get significant boosts:

```python
if context.user_intent == 'general':
    if hours_ago < 24:       # Last day: 8x boost
    elif hours_ago < 72:     # Last 3 days: 4x boost  
    elif hours_ago < 168:    # Last week: 2x boost
```

### 4. Two-Phase Search
For generic queries, first search recent memories:

```python
# Phase 1: Look in last 7 days
if is_generic_recall and not has_specificity:
    recent_memories = filter_by_date(memories, days=7)
    if recent_memories:
        use these first
```

## Examples of How It Works

### Example 1: DAGs
- User: "bring me back any DAG we have done"
- System: Detects generic pattern, adds recency focus, retrieves most recent DAG

### Example 2: SQL Queries
- User: "show me that query we wrote"
- System: Detects generic pattern, prioritizes recent SQL conversations

### Example 3: Documentation
- User: "can you bring back what we discussed?"
- System: Focuses on recent conversations regardless of topic

### Example 4: Specific Request (No Enhancement)
- User: "show me the query with the JOIN on users table"
- System: Has specificity, searches normally without recency bias

## Key Principles

1. **Domain Agnostic**: Works for any type of content
2. **User Intent Focused**: Understands when users want recent vs. specific
3. **Graceful Degradation**: Falls back to normal search if no recent matches
4. **Transparent**: Query enhancement is subtle and natural

## Testing the Solution

Test with various content types:
```
"bring me back any query we wrote" → Recent SQL
"show me that documentation we created" → Recent docs
"can you bring back what we discussed about authentication" → Specific topic
"that thing we were working on" → Most recent work
```

## Future Enhancements

1. **Learning User Patterns**: Track if user usually wants recent or old
2. **Context-Aware Recency**: Different recency windows for different topics
3. **Explicit Time Control**: "show me DAGs from last month"
4. **Memory Decay**: Automatic strength reduction over time