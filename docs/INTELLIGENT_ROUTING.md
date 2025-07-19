# Intelligent Model Routing in ADAM

ADAM now features intelligent model routing that automatically selects the best AI model for each query, optimizing for both performance and cost.

## 🎯 Model Hierarchy

### 1. **grok-4-reasoning** (High Complexity)
- **Use Cases**: Code generation, system design, complex debugging
- **Reasoning Effort**: High
- **Examples**:
  - "Write a distributed cache implementation"
  - "Design a scalable microservices architecture"
  - "Debug this complex async Python code"

### 2. **grok-4** (Medium Complexity)
- **Use Cases**: Analysis, explanations, standard queries
- **Reasoning Effort**: Medium
- **Examples**:
  - "Explain how neural networks work"
  - "Analyze this BigQuery query"
  - "What are best practices for API design?"

### 3. **grok-3-mini-high** (Low Complexity)
- **Use Cases**: Simple questions, memory recaps, quick lookups
- **Reasoning Effort**: Low
- **Examples**:
  - "What did we discuss yesterday?"
  - "List Python data types"
  - "Quick summary of our last conversation"

## 🧠 How It Works

### Query Analysis
ADAM analyzes each query for:
- **Complexity indicators**: Code-related keywords, deep analysis requests
- **Length**: Longer queries often need more sophisticated models
- **Context requirements**: Memory lookups vs. generation tasks
- **Technical depth**: Simple facts vs. complex reasoning

### Automatic Selection
```python
# Example: ADAM automatically routes queries
"What is Python?" → grok-3-mini-high (simple definition)
"Explain Python's GIL" → grok-4 (technical explanation)
"Implement a Python JIT compiler" → grok-4-reasoning (complex code)
```

## 💰 Cost Optimization

By using appropriate models for each task:
- Simple queries: Up to 90% cost savings
- Medium queries: Balanced cost/performance
- Complex queries: Maximum capability when needed

## 🔧 Manual Override

You can still specify a model explicitly:
```python
# Let ADAM choose (recommended)
response = await client.complete(query)

# Force a specific model
response = await client.complete(query, model="grok-4-reasoning")

# Force reasoning effort
response = await client.complete(query, reasoning_effort="high")
```

## 📊 Transparency

ADAM shows its decision-making:
```
[Query complexity: high, Using: grok-4-reasoning]
```

This helps you understand:
- Why a particular model was chosen
- The complexity assessment
- Cost implications

## 🚀 Benefits

1. **Cost Efficiency**: Automatically uses cheaper models when appropriate
2. **Performance**: Complex queries get the power they need
3. **Simplicity**: No need to manually select models
4. **Adaptability**: System learns query patterns over time

## 📝 Examples

### Code Generation (High Complexity)
```
Query: "Write a thread-safe LRU cache with TTL"
Model: grok-4-reasoning
Reasoning: Found code generation indicators
```

### Technical Explanation (Medium Complexity)
```
Query: "How does BigQuery handle partitioning?"
Model: grok-4
Reasoning: Technical analysis required
```

### Memory Recall (Low Complexity)
```
Query: "What was that SQL query from yesterday?"
Model: grok-3-mini-high
Reasoning: Simple memory lookup task
```

## 🔍 Query Analyzer API

For transparency, you can analyze any query:
```python
analysis = client.analyze_query("Your query here")
print(f"Complexity: {analysis['complexity']}")
print(f"Recommended: {analysis['recommended_model']}")
print(f"Reasoning: {analysis['reasoning']}")
```

## ⚙️ Configuration

The routing system is configured in:
- `src/adam/llm/config.py`: Model definitions
- `src/adam/llm/query_analyzer.py`: Analysis logic
- `src/adam/llm/client.py`: Routing implementation

## 🎭 Behind the Scenes

1. **Query Reception**: User submits query
2. **Complexity Analysis**: Keywords, length, patterns analyzed
3. **Model Selection**: Best model chosen based on requirements
4. **Effort Determination**: Reasoning effort set (low/medium/high)
5. **Execution**: Query processed with optimal settings
6. **Response**: User gets answer with model transparency

---

This intelligent routing ensures ADAM always uses the right tool for the job, balancing capability with efficiency.