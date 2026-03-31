# Sarcasm Detector

A tool for detecting sarcastic statements within longer narrative text documents. Analyzes paragraphs and multi-sentence passages to identify sarcasm with confidence scoring and contextual analysis. Created in 2 day hackathon

## Features

- **Document-Level Analysis**: Analyzes multi-paragraph texts, not just isolated sentences
- **Sarcasm Detection**: Identifies specific sarcastic statements within documents
- **Confidence Scoring**: Each detection includes a confidence score (0.0-1.0)
- **Context Awareness**: Shows surrounding sentences to understand why something is sarcastic
- **Sarcasm Classification**: Categorizes sarcasm types (emotional inversion, hyperbole, etc.)
- **Multi-Instance**: Finds multiple sarcastic statements in a single document
- **Context-Aware Explanations**: Explains each detection based on narrative context

## Example

**Input Document:**
```
Jessica and Sarah were going for a walk. Jessica was complaining about her manager at work. 
Jessica told Sarah how her manager smells and doesn't apply deodorant. "I'm not even sure 
he showers," said Jessica. Sarah replied, "oh come on, he has to shower. That would be gross." 
Jessica shook her head. "If he does shower," she said, "he needs a new body wash." 

Jessica also told Sarah that her manager doesn't seem to do anything at work. "He is always 
scrolling on Instagram and trying to show me what he thinks is funny," Jessica described. 
"It's always memes about cats. News flash, some of us are trying to work." 

Sarah replied, "he sounds like a real comedian". Jessica cringed and put her head in her hands. 
"I hope I can find a new job soon," she moaned.
```

**Detected Sarcasm:**

| Statement | Confidence | Type | Context |
|-----------|-----------|------|---------|
| "he sounds like a real comedian" | 94% | Emotional Inversion | After Jessica describes the manager as unfunny and disruptive |

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## Usage

### Via Streamlit UI

1. Go to the **Detector** tab
2. Select a model from the dropdown (3 options available)
3. Paste your document text (paragraph or longer)
4. Adjust the confidence threshold slider (0.5 is recommended)
5. Click **Detect Sarcasm**
6. Review detected instances with explanations and confidence scores

### Available Models

The app provides three model options:

1. **Default (Twitter-RoBERTa)** - `cardiffnlp/twitter-roberta-base-irony`
   - Pre-trained on Twitter irony detection
   - Good baseline, works well on general text
   - Accuracy: ~75% F1 score

2. **JSON Only** - `sennatitcomb/sarcasm-detector-json-only-final`
   - Fine-tuned on JSON/structured sarcasm data
   - Better precision for specific domains
   
3. **JSON + Joshi + Gutenberg** - `sennatitcomb/sarcasm-detector-json-joshi-gutenberg-final`
   - Fine-tuned on multiple datasets (JSON, Joshi, Gutenberg)
   - Most comprehensive training
   - Best for narrative and literary text

### Programmatically

```python
from src.document_sarcasm_detector import DocumentSarcasmDetector

# Initialize with default model
detector = DocumentSarcasmDetector(
    confidence_threshold=0.5,
    context_window=2
)

# Or initialize with a specific model
detector = DocumentSarcasmDetector(
    model_name="sennatitcomb/sarcasm-detector-json-joshi-gutenberg-final",
    confidence_threshold=0.5,
    context_window=2
)

# Analyze document
result = detector.detect_document(document_text)

# Access results
for instance in result.sarcasm_instances:
    print(f"Statement: {instance.sentence}")
    print(f"Confidence: {instance.confidence_score:.1%}")
    print(f"Type: {instance.sarcasm_type}")
    print(f"Explanation: {instance.explanation}")
    print()

# Print formatted report
print(detector.format_results(result))
```

---

## How It Works

### 1. Document Processing
- Input text is validated and cleaned
- Document is segmented into sentences using spaCy NLP

### 2. Sentence-Level Detection
- Each sentence is tokenized using the RoBERTa tokenizer
- Passed through the transformer model for classification
- Model outputs probabilities for sarcastic vs. non-sarcastic

### 3. Context Extraction
- Surrounding sentences (configurable window) are captured
- Provides narrative context for understanding the sarcasm

### 4. Sarcasm Classification
- Detected sarcasm is classified by type:
  - **Emotional Inversion**: Positive words for negative things
  - **Hyperbole**: Exaggeration markers
  - **Rhetorical Questions**: Question format
  - **General Sarcasm**: Other types

### 5. Result Assembly
- Confidence scores are normalized (0.0-1.0)
- Context-aware explanations are generated
- Results are filtered by threshold and sorted by confidence

---

## Configuration

### Confidence Threshold
- **Range**: 0.0 - 1.0
- **Default**: 0.5
- **Meaning**: Only show detections with confidence >= this value
- **Recommendation**: 0.5-0.7 for balanced precision/recall

### Context Window
- **Range**: 1-5 sentences
- **Default**: 2
- **Meaning**: Number of surrounding sentences to include as context
- **Recommendation**: 2-3 for good context coverage

---

## Model Details

### Base Model: `cardiffnlp/twitter-roberta-base-irony`

**Architecture**
- RoBERTa-base (110M parameters)
- Pre-trained on Twitter irony detection
- Fine-tuned for binary classification

**Capabilities**
- Input: Up to 512 tokens per sentence
- Output: Binary (sarcastic / non-sarcastic) + confidence
- Language: English

**Why This Model?**
- Pre-trained on irony/sarcasm detection task
- Good balance of accuracy and speed
- Transferable to narrative text with contextual analysis

---

## Output Format

### SarcasmInstance

```json
{
  "sentence": "he sounds like a real comedian",
  "sentence_index": 8,
  "confidence_score": 0.94,
  "context_before": [
    "Jessica told Sarah that her manager doesn't seem to do anything at work.",
    "He is always scrolling on Instagram and trying to show me what he thinks is funny."
  ],
  "context_after": ["Jessica cringed and put her head in her hands."],
  "sarcasm_type": "emotional_inversion",
  "explanation": "Sarah calls the manager 'a real comedian' after Jessica just described him as unfunny and disruptive. The contrast indicates sarcasm."
}
```

### DocumentSarcasmDetectionResult

```json
{
  "document_text": "...",
  "total_sentences": 15,
  "sarcasm_instances": [...],
  "confidence_threshold": 0.5,
  "sarcasm_count": 3,
  "sarcasm_percentage": 20.0
}
```

---

## Performance

### Accuracy
- Expected accuracy: ~80-85% on narrative text
- Varies based on sarcasm complexity and context clarity

### Speed
- **CPU**: ~2-3 seconds per document (100 words)
- **GPU**: ~200-400ms per document
- Scales linearly with document length

### Resource Requirements
- **RAM**: ~2GB for inference
- **GPU**: Optional (CUDA recommended for speed)
- **Disk**: ~500MB (model + dependencies)

---

## Limitations

1. **Context Dependency**: Works best when context clearly explains the sarcasm
2. **Subtle Sarcasm**: May miss very subtle or implicit sarcasm
3. **Language**: English only
4. **Sentence-Level**: Analyzes individual sentences (not multi-sentence sarcasm)
5. **Short Documents**: Requires sufficient text (100+ words recommended)

---

## Examples

### Example 1: Manager Complaint
The Jessica and Sarah example demonstrates sarcasm in dialogue with multiple instances of sarcasm within a narrative context.

**Key Detection**: "he sounds like a real comedian"
- **Why Sarcastic**: After Jessica describes the manager as unfunny and disruptive
- **Confidence**: 94%
- **Type**: Emotional inversion

### Example 2: Cooking Disaster
Shows sarcasm through positive comments about a terrible meal.

**Key Detections**:
- "this is absolutely delicious" (after burning food)
- "you've really outdone yourself as a chef"
- "you should open a restaurant"

---

## Files

```
src/
├── document_sarcasm_detector.py    # Main detector with context awareness
├── text_preprocessor.py            # spaCy NER preprocessing (optional)
└── fine_tune.py                    # Fine-tuning utilities

app.py                              # Streamlit UI
requirements.txt                    # Dependencies
notebooks/
└── fine_tune_guide.ipynb          # Fine-tuning guide (optional)
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- spaCy 3.5+
- Streamlit 1.28+
- Pydantic 2.0+

See `requirements.txt` for complete list.

---

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### CUDA Out of Memory
The detector processes one sentence at a time and is memory-efficient. If issues occur:
```python
detector = DocumentSarcasmDetector(device="cpu")
```

### Model Takes Long to Load
First run downloads the transformer model (~440MB). Subsequent runs use cached version.

### No Sarcasm Detected
- Lower the confidence threshold (try 0.3 instead of 0.5)
- Ensure text has sufficient context
- Check that document is in English

---

## Development

### Adding Custom Sarcasm Types
Edit `_detect_sarcasm_type()` in `document_sarcasm_detector.py` to add new classifications.

### Fine-tuning on Custom Data
See `notebooks/fine_tune_guide.ipynb` for complete fine-tuning workflow.

### Using Different Models
```python
detector = DocumentSarcasmDetector(
    model_name="distilbert-base-uncased"  # Any HF sequence classification model
)
```

---

## Citation

If you use this tool, please cite:

```bibtex
@software{document_sarcasm_detector,
  title={Document Sarcasm Detector},
  author={Senna Titcomb},
  year={2026},
  url={https://github.com/sennatitcomb/sarcasm-detector}
}
```

**Models Used:**
- Barbieri et al. (2020) - Twitter-RoBERTa-base-irony
  - https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
- Honnibal & Montani (2017) - spaCy
  - https://spacy.io

---

## License

This project is open source and available under the MIT License.

---

## Support

For issues, questions, or suggestions:
1. Check the "About" tab in the Streamlit app
2. Review examples in the "Examples" tab
3. Check troubleshooting section above

---

**Built with Hugging Face transformers, spaCy, PyTorch, and Streamlit**
