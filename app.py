"""
Streamlit UI for Document-Level Sarcasm Detection.

Analyzes longer text documents (paragraphs+) to identify sarcastic statements
within their narrative context.
"""

import streamlit as st
from pathlib import Path
import logging

from src.document_sarcasm_detector import DocumentSarcasmDetector, DocumentSarcasmDetectionResult

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Sarcasm Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Document Sarcasm Detector")
st.markdown(
    """
    Analyze longer text documents to identify and extract sarcastic statements 
    within their narrative context. Perfect for analyzing dialogue, narratives, and multi-paragraph text.
    """
)

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.header("Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        [
            "Default (Twitter-RoBERTa)",
            "JSON Only",
            "JSON + Joshi + Gutenberg"
        ],
        help="Choose which model to use for detection"
    )
    
    model_mapping = {
        "Default (Twitter-RoBERTa)": "cardiffnlp/twitter-roberta-base-irony",
        "JSON Only": "sennatitcomb/sarcasm-detector-json-only-final",
        "JSON + Joshi + Gutenberg": "sennatitcomb/sarcasm-detector-json-joshi-gutenberg-final"
    }
    
    selected_model = model_mapping[model_choice]
    
    st.divider()
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show sarcastic statements with confidence >= this value"
    )
    
    context_window = st.slider(
        "Context Window",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of surrounding sentences to show as context"
    )
    
    st.divider()
    
    page = st.radio(
        "Select Page",
        ["Detector", "Examples", "About"]
    )

# ============================================================================
# Initialize Model (Cached)
# ============================================================================

@st.cache_resource
def load_detector(model_name):
    """Load the document sarcasm detector."""
    return DocumentSarcasmDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        context_window=context_window
    )

# ============================================================================
# Page: Detector (Main)
# ============================================================================

if page == "Detector":
    st.header("Analyze Your Document")
    
    # Input area
    text_input = st.text_area(
        "Paste your document or narrative text:",
        height=250,
        placeholder="Enter a paragraph or longer text. Example:\n\nJessica and Sarah were going for a walk. Jessica was complaining about her manager..."
    )
    
    st.caption("💡 Works best with documents containing dialogue and narrative (100+ words)")
    
    # Detect button
    detect_btn = st.button("🔍 Detect Sarcasm", type="primary", use_container_width=True)
    
    # ========================================================================
    # Detection Logic
    # ========================================================================
    
    if detect_btn:
        if not text_input.strip():
            st.error("Please enter document text to analyze")
        else:
            with st.spinner("Analyzing document for sarcasm..."):
                try:
                    detector = load_detector(selected_model)
                    
                    # Detect
                    result: DocumentSarcasmDetectionResult = detector.detect_document(
                        text=text_input,
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Display results
                    st.divider()
                    st.header("Results")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Sentences", result.total_sentences)
                    
                    with col2:
                        st.metric("Sarcasm Detected", result.sarcasm_count)
                    
                    with col3:
                        st.metric("Sarcasm Density", f"{result.sarcasm_percentage:.1f}%")
                    
                    st.divider()
                    
                    # Detailed results
                    if result.sarcasm_instances:
                        st.success(f"✓ Found {result.sarcasm_count} sarcastic statement(s)")
                        
                        for i, instance in enumerate(result.sarcasm_instances, 1):
                            with st.expander(
                                f"#{i} - Confidence {instance.confidence_score:.0%} - {instance.sarcasm_type.replace('_', ' ').title()}",
                                expanded=(i == 1)
                            ):
                                # The sarcastic statement
                                st.markdown("**Sarcastic Statement:**")
                                st.info(f'"{instance.sentence}"')
                                
                                # Confidence score
                                confidence_col, type_col = st.columns(2)
                                with confidence_col:
                                    st.metric("Confidence Score", f"{instance.confidence_score:.1%}")
                                with type_col:
                                    st.metric("Type", instance.sarcasm_type.replace('_', ' ').title())
                                
                                # Context
                                if instance.context_before:
                                    st.markdown("**Context (Before):**")
                                    for j, ctx in enumerate(instance.context_before, 1):
                                        st.caption(f"_{j}. {ctx}_")
                                
                                if instance.context_after:
                                    st.markdown("**Context (After):**")
                                    for j, ctx in enumerate(instance.context_after, 1):
                                        st.caption(f"_{j}. {ctx}_")
                                
                                # Explanation
                                st.markdown("**Analysis:**")
                                st.write(instance.explanation)
                    else:
                        st.info(f"No sarcastic statements detected above {confidence_threshold:.0%} confidence threshold.")
                    
                    st.divider()
                    
                    # Formatted output
                    with st.expander("View Full Report"):
                        formatted = detector.format_results(result, include_context=True)
                        st.code(formatted)
                    
                    # JSON output
                    with st.expander("View JSON"):
                        import json
                        json_result = {
                            "total_sentences": result.total_sentences,
                            "sarcasm_count": result.sarcasm_count,
                            "sarcasm_percentage": result.sarcasm_percentage,
                            "instances": [inst.model_dump() for inst in result.sarcasm_instances]
                        }
                        st.json(json_result)
                    
                    # Store in session
                    st.session_state.last_result = result
            
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    logger.exception("Detection error")

# ============================================================================
# Page: Examples
# ============================================================================

elif page == "Examples":
    st.header("Example Documents")
    
    st.markdown("""
    Below are example documents with sarcasm. Try analyzing them to see how the detector works.
    """)
    
    # Example 1: Jessica & Sarah
    with st.expander("Example 1: Manager Complaint (Multiple Sarcastic Statements)"):
        example_text = """Jessica and Sarah were going for a walk. Jessica was complaining about her manager at work. Jessica told Sarah how her manager smells and doesn't apply deodorant. "I'm not even sure he showers," said Jessica. Sarah replied, "oh come on, he has to shower. That would be gross." Jessica shook her head. "If he does shower," she said, "he needs a new body wash." Jessica also told Sarah that her manager doesn't seem to do anything at work. "He is always scrolling on Instagram and trying to show me what he thinks is funny," Jessica described. "It's always memes about cats. News flash, some of us are trying to work." Sarah replied, "he sounds like a real comedian". Jessica cringed and put her head in her hands. "I hope I can find a new job soon," she moaned."""
        
        st.write(example_text)
        
        if st.button("Analyze Example 1"):
            st.session_state.example1_text = example_text
    
    # Example 2: Cooking
    with st.expander("Example 2: Cooking Disaster"):
        example_text2 = """Tom decided to cook dinner for his family. He had never cooked before. He burned the pasta, overcooked the chicken until it was dry as leather, and somehow managed to make the salad bitter. The kitchen was a disaster with dirty pans everywhere. His daughter took one bite and immediately reached for the trash can. His wife smiled politely and said, "Well, this is absolutely delicious! You've really outdone yourself as a chef." Tom beamed, thinking he had finally mastered the art of cooking. His son whispered to his sister, "I think we're ordering pizza tonight." Tom's wife continued, "I can't believe how talented you are in the kitchen. You should open a restaurant!" Tom felt proud, completely missing the sarcasm dripping from every word."""
        
        st.write(example_text2)
        
        if st.button("Analyze Example 2"):
            st.session_state.example2_text = example_text2
    
    # Process example if button was clicked
    if 'example1_text' in st.session_state:
        st.write("---")
        st.write("**Analyzing Example 1...**")
        with st.spinner("Processing..."):
            detector = load_detector(selected_model)
            result = detector.detect_document(st.session_state.example1_text, confidence_threshold)
            st.write(detector.format_results(result))
        del st.session_state.example1_text
    
    if 'example2_text' in st.session_state:
        st.write("---")
        st.write("**Analyzing Example 2...**")
        with st.spinner("Processing..."):
            detector = load_detector(selected_model)
            result = detector.detect_document(st.session_state.example2_text, confidence_threshold)
            st.write(detector.format_results(result))
        del st.session_state.example2_text

# ============================================================================
# Page: About
# ============================================================================

elif page == "About":
    st.header("About Document Sarcasm Detector")
    
    st.markdown("""
    ## Overview
    
    This tool detects sarcasm in longer text documents (narratives, dialogue, paragraphs) 
    and extracts sarcastic statements along with their narrative context.
    
    ### How It Works
    
    1. **Sentence Segmentation**: Documents are split into sentences using spaCy NLP
    2. **Sarcasm Detection**: Each sentence is analyzed using a transformer model 
       (cardiffnlp/twitter-roberta-base-irony)
    3. **Context Analysis**: Surrounding sentences are captured to understand why 
       something is sarcastic
    4. **Confidence Scoring**: Each detection has a confidence score (0.0-1.0)
    5. **Type Classification**: Sarcasm is classified (e.g., emotional inversion, hyperbole)
    
    ### Key Features
    
    - **Context Awareness**: Understands sarcasm by analyzing surrounding narrative
    - **Confidence Scoring**: Provides quantified certainty for each detection
    - **Sarcasm Typing**: Classifies types (emotional inversion, hyperbole, rhetorical questions)
    - **Multi-instance**: Finds multiple sarcastic statements in a single document
    - **Narrative Focus**: Designed for dialogue and narrative text (not just isolated sentences)
    
    ### Model Details
    
    **Base Model**: `cardiffnlp/twitter-roberta-base-irony`
    - Pre-trained on Twitter irony detection
    - RoBERTa-base architecture (110M parameters)
    - Binary classification (sarcastic / non-sarcastic)
    - Supports up to 512 tokens per sentence
    
    ### Example Use Cases
    
    - 📚 Analyzing dialogue in literature and scripts
    - 📰 Understanding sarcasm in written narratives
    - 🎬 Extracting sarcastic lines from screenplays
    - 💬 Analyzing conversational text for irony
    
    ### Limitations
    
    - Works best with paragraphs (100+ words)
    - Requires sufficient context to understand sarcasm
    - May miss subtle, context-dependent sarcasm
    - Language: English only
    
    ### How to Use
    
    1. Enter your document text in the "Detector" tab
    2. Adjust confidence threshold (0.5 is recommended)
    3. Click "Detect Sarcasm"
    4. Review detected instances with explanations
    
    ---
    
    **Built with**: Hugging Face 🤗 • spaCy 🪡 • PyTorch 🔥 • Streamlit ⚡
    """)

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Document Sarcasm Detector | Analyzes narrative text for sarcastic statements with context awareness
    </div>
    """,
    unsafe_allow_html=True
)
