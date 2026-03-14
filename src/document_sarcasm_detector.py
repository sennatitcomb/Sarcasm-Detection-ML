"""
Document-level sarcasm detector with context awareness.

Analyzes longer text documents (paragraphs+) to identify and extract 
sarcastic statements within their narrative context. Uses sentence-level
detection combined with contextual analysis.
"""

import re
from typing import List, Dict, Optional, Tuple
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class SarcasmInstance(BaseModel):
    """
    A detected instance of sarcasm in a document.
    
    Attributes:
        sentence: The sarcastic sentence (exact text)
        sentence_index: Which sentence in document (0-indexed)
        confidence_score: 0.0-1.0 confidence that this is sarcasm
        context_before: Preceding sentences that establish context (1-2 sentences)
        context_after: Following sentences (if available)
        sarcasm_type: Detected marker (e.g., "emotional_inversion", "irony")
        explanation: Why this is sarcastic based on context
    """
    sentence: str = Field(..., description="The sarcastic statement")
    sentence_index: int = Field(..., description="Position in document")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    context_before: List[str] = Field(default_factory=list, description="Previous sentences")
    context_after: List[str] = Field(default_factory=list, description="Following sentences")
    sarcasm_type: str = Field(default="sarcasm", description="Type of sarcasm detected")
    explanation: str = Field(default="", description="Why this is sarcastic")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentence": "he sounds like a real comedian",
                "sentence_index": 8,
                "confidence_score": 0.94,
                "context_before": [
                    "Jessica told Sarah that her manager doesn't seem to do anything at work.",
                    "He is always scrolling on Instagram and trying to show me what he thinks is funny."
                ],
                "context_after": ["Jessica cringed and put her head in her hands."],
                "sarcasm_type": "emotional_inversion",
                "explanation": "Sarah calls the manager 'a real comedian' after Jessica just described him as unfunny and disruptive. The contrast between 'real comedian' and the documented fact that he's not funny indicates sarcasm."
            }
        }


class DocumentSarcasmDetectionResult(BaseModel):
    """
    Complete result of document-level sarcasm detection.
    """
    document_text: str = Field(..., description="Original document")
    total_sentences: int = Field(..., description="Number of sentences in document")
    sarcasm_instances: List[SarcasmInstance] = Field(
        default_factory=list,
        description="Detected sarcasm instances (confidence >= threshold)"
    )
    confidence_threshold: float = Field(default=0.5)
    
    @property
    def sarcasm_count(self) -> int:
        """Number of sarcastic instances found."""
        return len(self.sarcasm_instances)
    
    @property
    def sarcasm_percentage(self) -> float:
        """Percentage of sentences that contain sarcasm."""
        if self.total_sentences == 0:
            return 0.0
        return (self.sarcasm_count / self.total_sentences) * 100


class DocumentSarcasmDetector:
    """
    Detects sarcasm in long-form text documents with context awareness.
    
    Key features:
    - Sentence-level segmentation with spaCy
    - Contextual analysis (preceding/following sentences)
    - Confidence scoring (0.0-1.0)
    - Sarcasm type classification
    - Context-aware explanations
    
    Example:
        >>> detector = DocumentSarcasmDetector()
        >>> result = detector.detect_document(long_paragraph_text)
        >>> for instance in result.sarcasm_instances:
        ...     print(f"{instance.sentence}")
        ...     print(f"Confidence: {instance.confidence_score:.1%}")
        ...     print(f"Explanation: {instance.explanation}")
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-irony",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        context_window: int = 2
    ):
        """
        Initialize the document sarcasm detector.
        
        Args:
            model_name: HF model identifier
            device: "auto" (GPU if available), "cuda", or "cpu"
            confidence_threshold: Only return instances >= this threshold
            context_window: Number of surrounding sentences to include as context
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.context_window = context_window
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            raise
        
        logger.info("✓ Models loaded successfully")
    
    def detect_document(
        self,
        text: str,
        confidence_threshold: Optional[float] = None
    ) -> DocumentSarcasmDetectionResult:
        """
        Detect sarcasm instances in a document.
        
        Args:
            text: Document text (can be multiple paragraphs)
            confidence_threshold: Override default threshold for this call
        
        Returns:
            DocumentSarcasmDetectionResult with detected instances
        """
        threshold = confidence_threshold or self.confidence_threshold
        
        # Validate input
        text = self._validate_input(text)
        
        # Segment document into sentences
        sentences = self._segment_sentences(text)
        
        if not sentences:
            return DocumentSarcasmDetectionResult(
                document_text=text,
                total_sentences=0,
                sarcasm_instances=[],
                confidence_threshold=threshold
            )
        
        # Detect sarcasm in each sentence
        sarcasm_instances = []
        
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
            
            # Predict sarcasm
            prediction = self._predict_sentence(sentence)
            confidence = prediction['confidence']
            
            # Include if above threshold
            if confidence >= threshold:
                # Get context
                context_before = sentences[max(0, i - self.context_window):i]
                context_after = sentences[i + 1:min(len(sentences), i + 1 + self.context_window)]
                
                # Generate explanation
                explanation = self._generate_explanation(
                    sentence=sentence,
                    context_before=context_before,
                    confidence=confidence
                )
                
                # Create instance
                instance = SarcasmInstance(
                    sentence=sentence.strip(),
                    sentence_index=i,
                    confidence_score=confidence,
                    context_before=[s.strip() for s in context_before],
                    context_after=[s.strip() for s in context_after],
                    sarcasm_type=self._detect_sarcasm_type(sentence),
                    explanation=explanation
                )
                
                sarcasm_instances.append(instance)
        
        # Sort by confidence (descending)
        sarcasm_instances.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return DocumentSarcasmDetectionResult(
            document_text=text,
            total_sentences=len(sentences),
            sarcasm_instances=sarcasm_instances,
            confidence_threshold=threshold
        )
    
    def _segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences using spaCy.
        
        Args:
            text: Document text
        
        Returns:
            List of sentences
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences
    
    def _predict_sentence(self, sentence: str) -> Dict:
        """
        Predict if a sentence contains sarcasm.
        
        Args:
            sentence: Single sentence to analyze
        
        Returns:
            Dict with 'is_sarcastic' (bool) and 'confidence' (float)
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(logits, dim=-1).item()
        
        # Model: 0=non-ironic, 1=ironic
        is_sarcastic = pred_class == 1
        confidence = probs[0, pred_class].item()
        
        return {
            'is_sarcastic': is_sarcastic,
            'confidence': confidence if is_sarcastic else (1.0 - confidence)
        }
    
    def _detect_sarcasm_type(self, sentence: str) -> str:
        """
        Heuristically detect type of sarcasm.
        
        Returns classification like "emotional_inversion", "hyperbole", etc.
        """
        sentence_lower = sentence.lower()
        
        # Positive words for negative context (emotional inversion)
        positive_words = [
            "great", "fantastic", "wonderful", "brilliant", "amazing", 
            "awesome", "excellent", "perfect", "beautiful", "lovely",
            "real", "sounds like", "must be"
        ]
        
        if any(word in sentence_lower for word in positive_words):
            return "emotional_inversion"
        
        # Hyperbole/exaggeration
        hyperbole_markers = ["always", "never", "everything", "nothing", "absolutely"]
        if any(word in sentence_lower for word in hyperbole_markers):
            return "hyperbole"
        
        # Rhetorical question
        if sentence.strip().endswith("?"):
            return "rhetorical_question"
        
        # Default
        return "sarcasm"
    
    def _generate_explanation(
        self,
        sentence: str,
        context_before: List[str],
        confidence: float
    ) -> str:
        """
        Generate explanation for why something is sarcastic based on context.
        
        Args:
            sentence: The sarcastic statement
            context_before: Preceding sentences
            confidence: Confidence score
        
        Returns:
            Explanation string
        """
        # Build context summary
        if context_before:
            context_summary = " ".join([s.strip()[:80] + "..." if len(s) > 80 else s.strip() 
                                       for s in context_before[-2:]])
        else:
            context_summary = "(no prior context)"
        
        explanation = (
            f"This statement appears sarcastic due to a contrast between the literal meaning "
            f"and the established context. After hearing: \"{context_summary}\", "
            f"the speaker's positive comment \"{sentence}\" is likely meant ironically. "
            f"(Confidence: {confidence:.0%})"
        )
        
        return explanation
    
    def _validate_input(self, text: str) -> str:
        """Validate and clean input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
        
        if len(text) > 20000:
            raise ValueError("Input text exceeds 20,000 characters")
        
        return text
    
    def format_results(
        self,
        result: DocumentSarcasmDetectionResult,
        include_context: bool = True
    ) -> str:
        """
        Format results as readable text.
        
        Args:
            result: Detection result
            include_context: Whether to include context sentences
        
        Returns:
            Formatted string
        """
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"Document Sarcasm Analysis",
            f"═══════════════════════════════════════════════════════════",
            f"Total sentences: {result.total_sentences}",
            f"Sarcastic statements found: {result.sarcasm_count}",
            f"Sarcasm density: {result.sarcasm_percentage:.1f}%",
            f"Confidence threshold: {result.confidence_threshold:.0%}",
            f"",
        ]
        
        if not result.sarcasm_instances:
            lines.append("No sarcastic statements detected above threshold.")
        else:
            for i, instance in enumerate(result.sarcasm_instances, 1):
                lines.extend([
                    f"─────────────────────────────────────────────────────────",
                    f"INSTANCE {i}",
                    f"─────────────────────────────────────────────────────────",
                    f"Sentence: \"{instance.sentence}\"",
                    f"Confidence: {instance.confidence_score:.1%}",
                    f"Type: {instance.sarcasm_type}",
                ])
                
                if include_context and instance.context_before:
                    lines.append(f"\nContext (before):")
                    for ctx in instance.context_before:
                        lines.append(f"  • {ctx}")
                
                lines.extend([
                    f"\nExplanation:",
                    f"  {instance.explanation}",
                    f""
                ])
        
        lines.append("═══════════════════════════════════════════════════════════")
        return "\n".join(lines)
