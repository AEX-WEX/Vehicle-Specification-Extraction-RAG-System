# Extractor Diagnostics Guide

This guide explains how to monitor and compare **Ollama vs Rule-Based extraction** performance, and check if Ollama is running.

## Quick Start

### 1. Check Ollama Status
```bash
python extract_diagnostics_cli.py status
```

Output:
```
================================================================================
OLLAMA STATUS
================================================================================

  Status: ✓ RUNNING
  URL:    http://localhost:11434

  ✓ Ollama is ready for extraction!

================================================================================
```

### 2. View Performance Report
After running several extractions, view the metrics:

```bash
python extract_diagnostics_cli.py report
```

Output:
```
================================================================================
EXTRACTOR DIAGNOSTICS REPORT
================================================================================

[Ollama Status]: ✓ Running

[Performance Metrics]:
--------------------------------------------------------------------------------

  OLLAMA:
    Total attempts:        10
    Successful:            8
    Failed:                2
    Success rate:          80.0%
    Avg specs extracted:   3.5
    Avg execution time:    245.3ms
    Avg confidence:        0.95

  RULE_BASED:
    Total attempts:        5
    Successful:            5
    Failed:                0
    Success rate:          100.0%
    Avg specs extracted:   2.1
    Avg execution time:    12.5ms
    Avg confidence:        0.70

[Recommendation]:
--------------------------------------------------------------------------------
  Best method:  OLLAMA
  Reasoning:    OLLAMA, ✓ Ollama success rate: 80.0%, ✓ Average specs extracted: 3.5
                  vs Rule-based: 2.1 specs, (Ollama available: True)
  Ollama:       Available

================================================================================
```

### 3. Get Recommendation
Get our recommendation on which method performs better:

```bash
python extract_diagnostics_cli.py recommend
```

## How It Works

### Architecture

```
Query to Extract
       ↓
SmartExtractor
       ↓
Ollama Available?
   Yes ↓ No
   ↓  └─→ Rule-Based ──→ Return Specs
   ↓                     (Track time & count)
Ollama Extract
   ↓
Success?
   ✓ ↓ ✗
   ↓ └─→ Fallback to Rule-Based
   ↓
Return Specs (Track time & count)
```

### Automatic Tracking

Every extraction automatically records:
- **Method used** (ollama, rule_based, or fallback)
- **Query** being processed
- **Number of specs** extracted
- **Execution time** (milliseconds)
- **Success/Failure** status
- **Confidence scores** for extracted specs
- **Ollama availability** at time of extraction

### Performance Scoring

The recommendation score is calculated:
```
Score = (Success Rate × 0.50) 
       + (Avg Specs / 5 × 0.30) 
       - (Execution Time / 1000 × 0.20)

Weights:
  - Success Rate:    50% importance (reliability)
  - Specs Extracted: 30% importance (thoroughness)
  - Speed:          -20% penalty   (slower = lower score)
```

## Integration in Your Code

### Option 1: Using SmartExtractor (Recommended)

```python
from src.extractor import SmartExtractor

# Initialize
extractor = SmartExtractor()

# Extract (automatically uses best method)
specs = extractor.extract(
    query="What is engine oil capacity?",
    contexts=[
        {
            "text": "Engine oil capacity: 4.5 liters",
            "chunk_id": "chunk_001",
            "page_number": 42
        }
    ]
)

# Get diagnostics
extractor.print_diagnostics()
```

### Option 2: Using OllamaExtractor Directly

```python
from src.extractor import OllamaExtractor

ollama = OllamaExtractor(model="llama3")
specs = ollama.extract(query, contexts)
# Automatically logged with method="ollama"
```

### Option 3: Using RuleBasedExtractor Directly

```python
from src.extractor import RuleBasedExtractor

rule_based = RuleBasedExtractor()
specs = rule_based.extract(contexts, query=query)
# Automatically logged with method="rule_based"
```

### Option 4: Direct Diagnostics Access

```python
from src.extractor_diagnostics import get_diagnostics

diagnostics = get_diagnostics()

# Get performance summary
summary = diagnostics.get_performance_summary()
print(summary)

# Get recommendation
rec = diagnostics.get_recommendation()
print(f"Best method: {rec['recommendation']}")

# Print full report
diagnostics.print_report()

# Export to CSV
csv_file = diagnostics.export_stats_csv("extraction_metrics.csv")
```

## CLI Commands

### Check Ollama Availability
```bash
python extract_diagnostics_cli.py status
```
- ✓ Shows if Ollama is running
- Shows Ollama server URL
- Suggests how to start Ollama if unavailable

### View Performance Report
```bash
python extract_diagnostics_cli.py report
```
- Shows metrics for each extraction method
- Success rates, average specs extracted, execution times
- Recommendation on which method to use

### Get Recommendation
```bash
python extract_diagnostics_cli.py recommend
```
- Recommends best extraction method
- Shows reasoning behind recommendation
- Displays detailed metrics comparison

### Export Metrics
```bash
python extract_diagnostics_cli.py export
```
- Exports all metrics to CSV file
- Location: `logs/extraction_stats.csv`
- Can be opened in Excel/spreadsheet applications

### Clear All Metrics
```bash
python extract_diagnostics_cli.py reset
```
- Clears all recorded metrics
- Starts fresh tracking
- Useful for clean comparison sessions

## Understanding the Metrics

### Success Rate
- Percentage of extractions that succeeded
- **Ollama typical**: 70-95% (depends on model quality)
- **Rule-based typical**: 80-100% (more deterministic)

### Specs Extracted
- Average number of specifications found per query
- **Ollama**: Often higher quality, more comprehensive
- **Rule-based**: More conservative, may miss complex specs

### Execution Time
- How long extraction took (in milliseconds)
- **Ollama**: 200-500ms (API call + LLM processing)
- **Rule-based**: 10-50ms (regex pattern matching)

### Confidence Score
- System's confidence in extracted specs (0.0 - 1.0)
- **Ollama**: 0.95+ (high confidence from LLM)
- **Rule-based**: 0.7 (conservative estimate)

## Troubleshooting

### Ollama Shows as "Not Running"
```bash
# Start Ollama
ollama serve

# Then in another terminal
python extract_diagnostics_cli.py status
# Should show ✓ RUNNING
```

### No Metrics Showing
```bash
# Run some extractions first
python main.py query "engine oil capacity"

# Then check diagnostics
python extract_diagnostics_cli.py report
```

### SmartExtractor Always Uses Rule-Based
1. Check if Ollama is available:
   ```bash
   python extract_diagnostics_cli.py status
   ```
2. If showing "✗ NOT RUNNING", start Ollama
3. If showing "✓ RUNNING", check logs for extraction errors

## Example Workflow

### Step 1: Test Ollama
```bash
$ python extract_diagnostics_cli.py status
[Ollama Status]: ✗ NOT RUNNING
```

### Step 2: Start Ollama (in separate terminal)
```bash
$ ollama serve
```

### Step 3: Verify Ollama Running
```bash
$ python extract_diagnostics_cli.py status
[Ollama Status]: ✓ RUNNING
```

### Step 4: Run Extractions
```bash
$ python main.py query "What is the brake fluid capacity?"
$ python main.py query "Engine oil type"
$ python main.py query "Tire pressure specifications"
```

### Step 5: Compare Performance
```bash
$ python extract_diagnostics_cli.py report
```

This shows you:
- How many specs each method found
- Success rates and execution times
- Which method is more effective

### Step 6: Get Recommendation
```bash
$ python extract_diagnostics_cli.py recommend
```

Returns the recommended method with scoring details.

## Logging

All diagnostics are automatically logged to:
- **JSON format**: `logs/extraction_stats.json`
- **CSV format**: `logs/extraction_stats.csv` (after export)

Each entry includes:
```json
{
  "timestamp": "2026-02-13T14:23:45.123456",
  "method": "ollama",
  "query": "engine oil capacity",
  "num_specs_extracted": 3,
  "execution_time_ms": 245.3,
  "success": true,
  "error_message": null,
  "ollama_available": true
}
```

## Best Practices

### 1. Let SmartExtractor Handle Selection
```python
# ✓ Good - uses best available method
from src.extractor import SmartExtractor
extractor = SmartExtractor()
specs = extractor.extract(query, contexts)
```

### 2. Check Ollama Availability
```bash
# Before running batch extractions
python extract_diagnostics_cli.py status
```

### 3. Monitor Performance Over Time
```bash
# Run after batch of extractions
python extract_diagnostics_cli.py report

# Track with version control
git add logs/extraction_stats.json
git commit -m "Add extraction metrics"
```

### 4. Export for Analysis
```bash
# After significant runs
python extract_diagnostics_cli.py export

# Open in Excel or data analysis tool
```

## Advanced: Custom Configuration

You can customize diagnostics location:

```python
from src.extractor_diagnostics import ExtractorDiagnostics

# Use custom stats file location
diagnostics = ExtractorDiagnostics(stats_file="my_custom_path/stats.json")

# Use it
diagnostics.record_extraction(
    method="ollama",
    query="test query",
    num_specs=5,
    execution_time_ms=250,
    success=True
)
```

## Summary

With these diagnostics tools, you can:
- ✅ **Check if Ollama is running** - `python extract_diagnostics_cli.py status`
- ✅ **Track which method is used** - Automatically logged per extraction
- ✅ **Compare performance** - `python extract_diagnostics_cli.py report`
- ✅ **Get recommendations** - `python extract_diagnostics_cli.py recommend`
- ✅ **Export metrics** - `python extract_diagnostics_cli.py export`
- ✅ **Fallback gracefully** - SmartExtractor handles failures

Choose **SmartExtractor** for production to get the best of both worlds!
