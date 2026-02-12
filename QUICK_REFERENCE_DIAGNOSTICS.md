# Quick Reference: Ollama vs Rule-Based Diagnostics

## Instant Check: Is Ollama Running?

```bash
python extract_diagnostics_cli.py status
```

**Output if running:**
```
Status: ✓ RUNNING
✓ Ollama is ready for extraction!
```

**Output if NOT running:**
```
Status: ✗ NOT RUNNING
✗ Ollama is not running.
Start it with: ollama serve
```

## One-Command Comparison Report

```bash
python extract_diagnostics_cli.py report
```

Shows:
- **Ollama** - specs found, success rate, speed, confidence
- **Rule-Based** - specs found, success rate, speed, confidence
- **Which is winning** right now

## Get Recommendation

```bash
python extract_diagnostics_cli.py recommend
```

Example output:
```
Best method: OLLAMA
Score: 0.82
Reasoning: OLLAMA, ✓ Ollama success rate: 83.3%
           vs Rule-based: 2.1 specs
```

## In Your Code: Use SmartExtractor

```python
from src.extractor import SmartExtractor

extractor = SmartExtractor()
specs = extractor.extract(query, contexts)
# Automatically chooses Ollama or falls back to rule-based
# And tracks performance!
```

## CLI Commands Cheat Sheet

| Command | Purpose |
|---------|---------|
| `python extract_diagnostics_cli.py status` | Check if Ollama is running |
| `python extract_diagnostics_cli.py report` | See performance metrics for both methods |
| `python extract_diagnostics_cli.py recommend` | Which method is performing better? |
| `python extract_diagnostics_cli.py export` | Save metrics to CSV |
| `python extract_diagnostics_cli.py reset` | Clear all metrics |

## Understanding the Metrics

### Success Rate
- Percentage of successful extractions
- **Ollama**: Usually 70-95%
- **Rule-Based**: Usually 80-100%
- **Winner**: Higher % = better

### Specs Extracted
- Average number of specs found per query
- **Ollama**: Often finds more (more intelligent)
- **Rule-Based**: May find fewer (pattern-matching only)
- **Winner**: More specs = better

### Execution Time
- How fast (milliseconds)
- **Ollama**: 200-500ms (LLM processing)
- **Rule-Based**: 10-50ms (fast regex)
- **Winner**: Faster is better

### Confidence Score
- System's confidence (0.0 - 1.0)
- **Ollama**: 0.95+ (high confidence from LLM)
- **Rule-Based**: 0.7 (conservative estimate)
- **Winner**: Higher = more reliable

## What Happens Behind the Scenes?

```
Your Query
    ↓
SmartExtractor
    ↓
Is Ollama running?
    ├─ YES → Try Ollama
    │         ├─ Success? ✓ Return specs (logged as "ollama")
    │         └─ Failed? → Fallback to rule-based
    │                       (logged as "fallback")
    │
    └─ NO  → Use rule-based
             Return specs (logged as "rule_based")
    ↓
All metrics automatically saved to:
  logs/extraction_stats.json
```

## Example Workflow

### 1. Start Fresh
```bash
python extract_diagnostics_cli.py reset
echo "Metrics cleared"
```

### 2. Run Some Extracting
```bash
# Run several queries through the system
python main.py query "Engine oil capacity"
python main.py query "Brake torque specs"
python main.py query "Tire pressure"
# etc...
```

### 3. Check Status
```bash
python extract_diagnostics_cli.py status
```

### 4. Compare Performance
```bash
python extract_diagnostics_cli.py report
```

Sample output:
```
OLLAMA:
  Total attempts:       10
  Success rate:         80.0%
  Avg specs:            3.5
  Avg time:             245ms

RULE_BASED:
  Total attempts:       2
  Success rate:         100.0%
  Avg specs:            2.1
  Avg time:             12ms
```

### 5. Get Recommendation
```bash
python extract_diagnostics_cli.py recommend
```

Output: `Best method: OLLAMA (Score: 0.78)`

## Performance Scoring Formula

```
Score = (Success Rate × 50%) 
       + (Specs/5 × 30%)
       - (Time/1000 × 20%)

Example:
  Ollama:     (0.80 × 0.50) + (3.5/5 × 0.30) - (245/1000 × 0.20)
            = 0.40 + 0.21 - 0.049 = 0.561

  Rule-based: (1.00 × 0.50) + (2.1/5 × 0.30) - (12/1000 × 0.20)
            = 0.50 + 0.126 - 0.002 = 0.624
```

## Data Stored

Metrics saved in: `logs/extraction_stats.json`

Each entry:
```json
{
  "timestamp": "2026-02-13T14:23:45",
  "method": "ollama",
  "query": "engine oil capacity",
  "num_specs_extracted": 3,
  "execution_time_ms": 245,
  "success": true,
  "ollama_available": true
}
```

## Troubleshooting

### Q: Status shows "✗ NOT RUNNING" but Ollama should be running
**A:** 
```bash
# Restart Ollama
ollama serve

# Then test again
python extract_diagnostics_cli.py status
```

### Q: Rules constantly win despite using OllamaExtractor
**A:**
- Check logs: `tail -f logs/extraction_stats.json`
- Verify Ollama model is loaded: `ollama list`
- Try a different model: `OllamaExtractor(model="mistral")`

### Q: No metrics showing even after many queries
**A:**
```bash
# Make sure you're using SmartExtractor
python -c "from src.extractor import SmartExtractor; \
           e = SmartExtractor(); \
           specs = e.extract('test', [{'text': 'test', 'chunk_id': 'c1', 'page_number': 1}]); \
           print(len(specs))"

# Then check
python extract_diagnostics_cli.py report
```

## Key Insights

### When Ollama Wins
- Complex queries requiring understanding
- Technical specifications needing interpretation
- Quality is worth 50x speed penalty

### When Rule-Based Wins
- Simple pattern extraction is sufficient
- Speed is critical (real-time systems)
- No internet connection available
- Low-memory environments

### Smart Strategy
**Use SmartExtractor** - it automatically:
1. Checks if Ollama is available
2. Tries Ollama for best quality
3. Falls back to rule-based if needed
4. Logs everything for comparison
5. Learns which performs better

## Summary

- **Check Ollama**: `python extract_diagnostics_cli.py status`
- **Compare Methods**: `python extract_diagnostics_cli.py report`
- **Get Recommendation**: `python extract_diagnostics_cli.py recommend`
- **Use in Code**: `SmartExtractor().extract(query, contexts)`
- **Metrics Location**: `logs/extraction_stats.json`

---

**Next Step**: Run your query and check which method wins!
