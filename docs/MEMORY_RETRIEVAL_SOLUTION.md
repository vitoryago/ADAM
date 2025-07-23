# ADAM Memory Retrieval Solution

## The Problem

When users ask generic questions like "bring me back any DAG we have done?", ADAM fails to retrieve recent conversations and instead returns old, generic DAG examples. This happens even though the recent conversation IS properly saved in memory.

### Root Cause Analysis

1. **Memory Competition**: Older memories have very high strength scores (0.95-1.00) from repeated access
2. **Generic Query Matching**: Generic queries match MANY old conversations equally well
3. **Insufficient Recency Bias**: Current timestamp boosting (5x) isn't enough to overcome strength differences
4. **No Query Context**: Generic queries don't include temporal hints that would trigger recency boosting

### Example
- Today's DAG memory: strength 0.72, timestamp 2025-07-23
- Old generic DAG memory: strength 0.98, timestamp 2025-07-19
- Even with 5x boost: 0.72 * 5 = 3.6 < 0.98 * similarity_score

## The Solution

### 1. Enhanced Web Interface Memory Search

Update `web/adam_web.py` to add temporal context to generic queries:

```python
# In process_message method, before memory search:

# Add temporal hints to generic queries
if st.session_state.get('use_memory', True):
    # Check if query is generic and about past conversations
    query_lower = prompt.lower()
    generic_patterns = [
        "any dag", "some dag", "a dag", 
        "bring me back", "show me", "can you"
    ]
    
    if any(pattern in query_lower for pattern in generic_patterns):
        # Add recency hint to improve retrieval
        if "recent" not in query_lower and "last" not in query_lower:
            enhanced_query = f"{prompt} (focusing on our recent conversations)"
        else:
            enhanced_query = prompt
    else:
        enhanced_query = prompt
```

### 2. Two-Phase Memory Search Strategy

Implement a two-phase search that prioritizes recent memories:

```python
# Phase 1: Search recent memories (last 7 days)
recent_memories = []
all_memories = st.session_state.memory.recall_with_context(
    query=enhanced_query,
    n_results=20  # Get more candidates
)

# Filter for recent memories
from datetime import datetime, timedelta
cutoff_date = datetime.now() - timedelta(days=7)

for memory in all_memories:
    timestamp_str = memory.get('metadata', {}).get('timestamp', '')
    if timestamp_str:
        try:
            memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if memory_time > cutoff_date:
                recent_memories.append(memory)
        except:
            pass

# Use recent memories if found, otherwise fall back to all
memories_to_use = recent_memories[:5] if recent_memories else all_memories[:5]
```

### 3. Explicit Recency Instructions in System Prompt

Update the system prompt to prioritize recent conversations:

```python
system_prompt = """You are ADAM, an AI assistant with perfect memory. 

CRITICAL INSTRUCTIONS:
1. When users ask about "any DAG" or use generic references, they usually mean 
   the MOST RECENT one you discussed together.
2. Always check the timestamp of memories and prioritize recent conversations.
3. When multiple memories match, choose the most recent one unless the user 
   specifically asks for older examples.
4. If you find multiple DAG conversations, explicitly mention you're showing 
   the most recent one.
"""
```

### 4. Memory Strength Decay (Long-term Solution)

Implement automatic strength decay for older memories:

```python
# In memory_lifecycle.py
def apply_time_based_decay(self):
    """Reduce strength of old memories to favor recent ones"""
    for memory_id, metadata in self.memories.items():
        timestamp = metadata.get('timestamp')
        if timestamp:
            age_days = (datetime.now() - timestamp).days
            if age_days > 30:
                # Reduce strength by 10% per month
                decay_factor = 0.9 ** (age_days / 30)
                metadata['strength'] *= decay_factor
```

## Implementation Steps

1. **Immediate Fix**: Update `adam_web.py` with query enhancement
2. **Better Fix**: Implement two-phase search strategy
3. **Best Fix**: Add memory decay to naturally favor recent conversations

## Testing

After implementation, test with these queries:
- "bring me back any DAG we have done" → Should return today's DAG
- "show me the DAG we discussed" → Should return today's DAG
- "what DAGs have we created?" → Should list recent ones first

## Expected Outcome

Users will get the DAG they're actually looking for (the recent one) instead of generic examples from old conversations. This matches user expectations that ADAM remembers their recent work.