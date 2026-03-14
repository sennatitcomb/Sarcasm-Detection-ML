# Quick Start Guide

## Installation

1. Create Python 3.12 venv:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import spacy; import subprocess; subprocess.run(['uv', 'pip', 'install', 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'])"
```

## Using the Application

### Via Streamlit Web UI

```bash
streamlit run app.py
```

Then visit `http://localhost:8501` in your browser.

Features:
- Paste document text (any length)
- Adjust confidence threshold (0.0-1.0, default 0.5)
- Adjust context window (1-5 sentences, default 2)
- See sarcasm instances with confidence scores and explanations

### Programmatically

```python
from src.document_sarcasm_detector import DocumentSarcasmDetector

detector = DocumentSarcasmDetector(
    confidence_threshold=0.5,
    context_window=2
)

text = """Your text here..."""

result = detector.detect_document(text)

# Get results
print(f"Found {result.sarcasm_count} instances of sarcasm")
for instance in result.sarcasm_instances:
    print(f"  - {instance.sentence}")
    print(f"    Confidence: {instance.confidence_score:.1%}")
    print(f"    Type: {instance.sarcasm_type}")
```

## Testing the Example

Try the Jessica and Sarah manager complaint example:

```python
from src.document_sarcasm_detector import DocumentSarcasmDetector

detector = DocumentSarcasmDetector(confidence_threshold=0.5)

text = """Jessica and Sarah were going for a walk. Jessica was complaining about her manager at work. 
Jessica told Sarah how her manager smells and doesn't apply deodorant. "I'm not even sure 
he showers," said Jessica. Sarah replied, "oh come on, he has to shower. That would be gross." 
Jessica shook her head. "If he does shower," she said, "he needs a new body wash." 

Jessica also told Sarah that her manager doesn't seem to do anything at work. "He is always 
scrolling on Instagram and trying to show me what he thinks is funny," Jessica described. 
"It's always memes about cats. News flash, some of us are trying to work." 

Sarah replied, "he sounds like a real comedian". Jessica cringed and put her head in her hands. 
"I hope I can find a new job soon," she moaned."""

result = detector.detect_document(text)
print(detector.format_results(result))
```

Expected output:
- Detects "he sounds like a real comedian" at ~87% confidence
- Detects other sarcastic instances with context

## Fine-Tuning for Better Results

To improve the model on your specific data:

1. See [COLAB_FINE_TUNING.md](COLAB_FINE_TUNING.md)
2. Run fine-tuning in Google Colab (takes ~20 minutes)
3. Download the fine-tuned model
4. Update detector to use new model:

```python
detector = DocumentSarcasmDetector(
    model_name="path/to/finetuned/model"
)
```

## Project Structure

```
src/
├── document_sarcasm_detector.py    Main detector class
└── __init__.py

app.py                             Streamlit UI
requirements.txt                   Dependencies
README.md                          Full documentation
COLAB_FINE_TUNING.md             Fine-tuning guide
QUICK_START.md                    This file
```

## Troubleshooting

**Error: No module named 'spacy'**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Error: Model not found**
```bash
# First run will download model (~500MB), be patient
```

**Model takes too long**
- Use GPU if available: CUDA-enabled PyTorch
- Or reduce document length (split into smaller passages)

**Too many false positives**
- Increase confidence threshold (try 0.6 or 0.7)
- Fine-tune on domain-specific data (see COLAB_FINE_TUNING.md)

**Not enough detections**
- Lower confidence threshold (try 0.3 or 0.4)
- Ensure document has sufficient context
- Check document is in English
