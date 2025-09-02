# ADAM Knowledge Base Repository

## Overview
This repository contains structured knowledge bases that enhance ADAM's expertise in specific domains. Each knowledge base is intelligently accessed only when relevant context is detected, ensuring focused and efficient assistance.

## Directory Structure

```
knowledge/
├── dbt/                    # Data Build Tool (DBT) knowledge
│   ├── README.md                # DBT integration documentation
│   ├── DBT_KNOWLEDGE.md         # Comprehensive DBT best practices
│   └── dbt_patterns.yaml        # Structured DBT patterns and templates
│
├── sql/                    # SQL optimization and query knowledge
│   ├── README.md                      # SQL integration documentation
│   ├── SNOWFLAKE_SQL_KNOWLEDGE.md    # Snowflake SQL best practices
│   └── snowflake_sql_patterns.yaml   # SQL patterns and optimization rules
│
└── looker_dbt_migration/   # Looker PDT to DBT migration
    ├── README.md                             # Migration guide
    ├── LOOKER_DBT_MIGRATION_KNOWLEDGE.md    # Comprehensive migration patterns
    └── looker_dbt_patterns.yaml             # Structured conversion templates
```

## Knowledge Domains

### 1. DBT Knowledge (`/dbt/`)
**Purpose**: Expert guidance for Data Build Tool (dbt) development

**Features**:
- Layer architecture patterns (Staging → Intermediate → Marts)
- Naming conventions and folder structure
- Incremental materialization strategies
- Snowflake-specific optimizations
- Testing and documentation patterns
- Macro templates and utilities

**Activation**: Intelligently detected using Claude Haiku when queries involve:
- DBT model creation or conversion
- Data transformation pipelines
- Analytics engineering
- Looker PDT migrations
- Data warehouse modeling

**Files**:
- `DBT_KNOWLEDGE.md`: 12.8KB comprehensive guide
- `dbt_patterns.yaml`: 15.2KB structured patterns

### 2. SQL Knowledge (`/sql/`)
**Purpose**: High-performance SQL query optimization for Snowflake

**Features**:
- UPPERCASE keyword enforcement
- Query optimization patterns
- Anti-pattern detection and fixes
- Snowflake-specific optimizations
- Performance monitoring queries
- Safe operation templates

**Activation**: Intelligently detected when queries involve:
- SQL query writing or optimization
- Database performance tuning
- Query formatting and standards
- Snowflake features
- Table design and indexing

**Files**:
- `SNOWFLAKE_SQL_KNOWLEDGE.md`: 14.3KB optimization guide
- `snowflake_sql_patterns.yaml`: 15KB structured patterns

### 3. Looker-to-DBT Migration (`/looker_dbt_migration/`)
**Purpose**: Convert Looker PDTs to DBT models with backward compatibility

**Features**:
- PDT analysis and classification
- Automatic DBT model generation
- Looker view generation for DBT models
- Incremental conversion strategies
- Performance optimization during migration
- Validation query generation

**Activation**: Detected when queries involve:
- Looker PDT conversion
- LookML to DBT migration
- Derived table optimization
- PDT to incremental model conversion
- Looker-DBT integration

**Files**:
- `LOOKER_DBT_MIGRATION_KNOWLEDGE.md`: Comprehensive migration patterns
- `looker_dbt_patterns.yaml`: Structured conversion templates
- `README.md`: Migration guide and examples


## Intelligent Access System

### Detection Method
ADAM uses a two-tier detection system:

1. **Primary**: Claude Haiku LLM for intelligent context detection
   - Analyzes query intent and context
   - Returns confidence scores (0.0-1.0)
   - Activates knowledge when confidence > 0.3

2. **Fallback**: Keyword-based detection
   - Used when LLM detection fails
   - Checks for domain-specific keywords
   - Ensures knowledge is always available when needed

### Integration Points

**Service Classes**:
- `DBTKnowledgeService` (`/src/adam_v2/services/dbt_knowledge_service.py`)
- `SQLKnowledgeService` (`/src/adam_v2/services/sql_knowledge_service.py`)

**Main Integration**:
- `LLMService` (`/src/adam_v2/services/llm_service.py`)
  - Calls detection methods
  - Enhances queries with relevant context
  - Manages knowledge activation

## Adding New Knowledge Bases

To add a new knowledge domain:

1. **Create Directory Structure**:
   ```bash
   mkdir knowledge/your_domain
   ```

2. **Add Knowledge Files**:
   - Markdown file with comprehensive knowledge
   - YAML file with structured patterns
   - Examples and templates

3. **Create Service Class**:
   ```python
   class YourKnowledgeService:
       def __init__(self):
           self.knowledge_base_path = Path(__file__).parent.parent.parent.parent / "knowledge"
           self.patterns_file = self.knowledge_base_path / "your_domain" / "patterns.yaml"
           self.knowledge_file = self.knowledge_base_path / "your_domain" / "knowledge.md"
   ```

4. **Implement Detection**:
   - Add `intelligent_detection()` method using Claude Haiku
   - Add `detect_context()` fallback method
   - Set confidence thresholds

5. **Integrate with LLMService**:
   - Import your service class
   - Add detection logic to `get_completion()`
   - Enhance queries when context detected

## Self-Learning Capabilities

Both DBT and SQL knowledge services include memory integration for continuous improvement:

- **Pattern Storage**: Successful patterns saved with quality scores
- **Similarity Search**: Retrieves relevant past solutions
- **Performance Tracking**: Monitors execution time and efficiency
- **Quality Metrics**: Tracks test results and optimization scores

## Usage Examples

### DBT Query
```
User: "Create a staging model for the orders table"
ADAM: [Detects DBT context with 0.95 confidence]
      [Loads DBT knowledge and patterns]
      [Generates model with proper conventions]
```

### SQL Query
```
User: "Write a query to find top customers by revenue"
ADAM: [Detects SQL context with 0.88 confidence]
      [Loads SQL optimization rules]
      [Generates query with UPPERCASE keywords and optimizations]
```

### Non-Domain Query
```
User: "How do I center a div in CSS?"
ADAM: [No domain knowledge activated]
      [Responds with general assistance]
```

## Performance Considerations

- Knowledge files are loaded once on service initialization
- Total knowledge size: ~60KB (minimal memory impact)
- LLM detection adds ~100ms latency (acceptable for quality)
- Fallback detection is instantaneous
- Context enhancement typically adds 500-2000 tokens

## Maintenance

### Updating Knowledge
1. Edit markdown/YAML files directly
2. Restart ADAM backend to reload
3. Test with relevant queries

### Monitoring
- Check logs for detection confidence scores
- Review enhanced query sizes
- Monitor memory pattern quality scores

### Version Control
- All knowledge files are tracked in Git
- Changes are versioned with the codebase
- Knowledge evolves with project requirements

## Benefits

✅ **Specialized Expertise**: Deep knowledge in specific domains  
✅ **Intelligent Activation**: Only loads relevant context  
✅ **Self-Improving**: Learns from successful patterns  
✅ **Maintainable**: Organized structure for easy updates  
✅ **Performant**: Minimal overhead with maximum value  

## Future Enhancements

Planned improvements:
- Additional knowledge domains (Terraform, Kubernetes, AWS)
- Knowledge versioning system
- A/B testing for pattern effectiveness
- Automatic knowledge extraction from documentation
- Knowledge graph relationships between domains

---

*Knowledge is power, but organized knowledge is a superpower.* 🚀