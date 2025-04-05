"""
Streamlit web app for the Text Humanizer.
"""
import os
import sys
import streamlit as st
import time
import base64
import json
import uuid
import html
from pathlib import Path

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from the humanizer package
from humanizer import TextHumanizer
from humanizer.utils.backup import create_backup, restore_backup
from humanizer.llm import LLMInterface, get_available_models
from humanizer.llm import get_prompt_template, list_available_prompts

# Set page configuration
st.set_page_config(
    page_title="Text Humanizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0277BD;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-area {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        background-color: #F5F5F5;
    }
    .info-box {
        background-color: #1e2931;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #2f5b32;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
    }
    .copy-btn {
        background-color: #1E88E5;
        color: white;
        padding: 8px 12px;
        border: none;
        border-radius: 4px;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .copy-btn:hover {
        background-color: #1565C0;
    }
    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 80px;
        background-color: #555;
        color: #fff;
        text-align: center;
        padding: 5px 0;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -40px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    /* Show the tooltip text when hovering */
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a download link for text
def get_download_link(text, filename, link_text):
    """Generate a link to download text content as a file"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
    return href

# Replace complex copy button functions with a simpler approach
def show_copyable_text(text):
    """Display text in a way that's easy to copy in Streamlit"""
    st.code(text, language=None)
    st.info("👆 Click in the box above, press Ctrl+A to select all, then Ctrl+C to copy")

# Create sidebar for configuration
st.sidebar.markdown("""
<div class="main-header">Text Humanizer</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("Make your text more human-readable and bypass AI detection!")

# Path for saved models
MODELS_DIR = os.path.join(current_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize session state
if 'humanizer' not in st.session_state:
    st.session_state.humanizer = TextHumanizer()

if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None

if 'humanized_text' not in st.session_state:
    st.session_state.humanized_text = ""
    
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None

if 'llm_interface' not in st.session_state:
    st.session_state.llm_interface = None

if 'llm_connected' not in st.session_state:
    st.session_state.llm_connected = False

# Function to train the model
def train_model(texts, model_name):
    with st.spinner("Training style model..."):
        # Train the humanizer on the provided texts
        st.session_state.humanizer.train_on_texts(texts)
        
        # Save the model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        st.session_state.humanizer.save_style_model(model_path)
        st.session_state.loaded_model = model_name
        
        return True

# Tabs for different functionalities - removed the backup tab
tab1, tab2, tab3 = st.tabs(["Humanize Text", "Train Style Model", "LLM Integration"])

# Tab 1: Humanize Text
with tab1:
    st.markdown('<div class="main-header">Humanize Your Text</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Transform your text to make it more natural and human-like. The humanizer can simplify complex vocabulary,
        correct grammar, and apply your custom writing style.
    </div>
    """, unsafe_allow_html=True)
    
    # Text input
    input_text = st.text_area(
        "Enter the text you want to humanize:",
        height=200,
        placeholder="Paste your text here..."
    )
    
    # Configuration options in an expander
    with st.expander("Configuration Options", expanded=True):
        # Two columns layout for options
        col1, col2 = st.columns(2)
        
        with col1:
            # Add LLM option if we have a connected LLM
            use_llm = st.checkbox(
                "Use LLM for Humanization", 
                value=st.session_state.llm_connected,
                disabled=not st.session_state.llm_connected,
                help="Use connected LLM model for more advanced humanization"
            )
            
            # Show prompt selection only if LLM is selected
            if use_llm and st.session_state.llm_connected:
                # Get available prompt descriptions
                prompt_descriptions = list_available_prompts()
                prompt_options = list(prompt_descriptions.keys())
                
                # Format for display
                prompt_format = lambda x: f"{x.capitalize()}: {prompt_descriptions[x]}"
                
                selected_prompt = st.selectbox(
                    "Humanization Style",
                    options=prompt_options,
                    format_func=prompt_format,
                    help="Select the style of humanization to apply"
                )
                
                # Replace nested expander with checkbox to avoid nesting violation
                show_prompt = st.checkbox("Show selected prompt template", value=False)
                if show_prompt:
                    st.code(get_prompt_template(selected_prompt).replace("{text}", "[Your text here]"))
            else:
                # Original options
                correct_grammar = st.checkbox("Correct Grammar", value=True, 
                                             help="Fix grammatical errors in the text")
                simplify_vocab = st.checkbox("Simplify Vocabulary", value=True,
                                            help="Replace complex words with simpler alternatives")
                
                # Style model selection
                model_files = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
                if model_files:
                    selected_model = st.selectbox(
                        "Select a style model:",
                        options=model_files,
                        index=model_files.index(st.session_state.loaded_model) if st.session_state.loaded_model in model_files else 0,
                        help="Choose a trained style model to apply"
                    )
                    
                    if st.button("Load Selected Model"):
                        model_path = os.path.join(MODELS_DIR, f"{selected_model}.pkl")
                        with st.spinner("Loading model..."):
                            if st.session_state.humanizer.load_style_model(model_path):
                                st.session_state.loaded_model = selected_model
                                st.success(f"Loaded model: {selected_model}")
                            else:
                                st.error("Failed to load model")
                else:
                    st.info("No style models available. Train a model in the 'Train Style Model' tab.")
        
        with col2:
            word_length = st.slider(
                "Word length threshold", 
                min_value=4, 
                max_value=15, 
                value=8,
                help="Words longer than this will be simplified"
            )
            
            apply_style = st.checkbox(
                "Apply Style", 
                value=True if st.session_state.loaded_model else False,
                disabled=st.session_state.loaded_model is None,
                help="Apply your trained style (requires a trained model)"
            )
            
            style_strength = st.slider(
                "Style Strength", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7,
                step=0.1,
                help="How strongly to apply the style (higher = more aggressive)"
            )
    
    # Process button with a more attractive design
    process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
    with process_col2:
        process_button = st.button("✨ Humanize Text", use_container_width=True, type="primary")
    
    # Process text when button is clicked
    if process_button:
        if not input_text.strip():
            st.warning("Please enter some text to humanize.")
        else:
            with st.spinner("Processing text..."):
                start_time = time.time()
                
                if use_llm and st.session_state.llm_connected and st.session_state.llm_interface:
                    # Get the selected prompt template
                    prompt_template = get_prompt_template(selected_prompt)
                    
                    # Use LLM-based humanization
                    humanized_text = st.session_state.humanizer.humanize_with_llm(
                        input_text,
                        st.session_state.llm_interface,
                        prompt_template=prompt_template,
                        temperature=0.7 if selected_prompt != "avoid_detection" else 0.85
                    )
                else:
                    # Use traditional humanization
                    humanized_text = st.session_state.humanizer.humanize(
                        input_text,
                        simplify=simplify_vocab, 
                        correct=correct_grammar,
                        word_length_threshold=word_length,
                        apply_style=apply_style,
                        style_strength=style_strength
                    )
                
                # Store the result in session state
                st.session_state.humanized_text = humanized_text
                
                # Store processing time in session state
                st.session_state.processing_time = time.time() - start_time
            
            # Show success message
            st.markdown("""
            <div class="success-box">
                Text has been successfully humanized! You can compare the original and humanized versions below.
            </div>
            """, unsafe_allow_html=True)
    
    # Display results if we have humanized text
    if st.session_state.humanized_text:
        # Display the side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Original Text</div>', unsafe_allow_html=True)
            st.text_area("", input_text, height=250, disabled=True, label_visibility="collapsed")
            
        with col2:
            st.markdown('<div class="sub-header">Humanized Text</div>', unsafe_allow_html=True)
            # Text area for the humanized text
            humanized_output = st.text_area(
                "", 
                st.session_state.humanized_text, 
                height=250, 
                key="humanized_output", 
                label_visibility="collapsed"
            )
            
            # Copy and download options
            copy_tab, download_tab = st.tabs(["Copy Text", "Download"])
            
            with copy_tab:
                st.markdown("### Easy Copy")
                show_copyable_text(st.session_state.humanized_text)
                
            with download_tab:
                st.markdown("### Download as File")
                st.markdown(
                    get_download_link(
                        st.session_state.humanized_text, 
                        "humanized_text.txt", 
                        "📥 Download Text as File"
                    ),
                    unsafe_allow_html=True
                )
        
        # Display statistics if processing time is available
        if st.session_state.processing_time is not None:
            st.caption(f"Processing time: {st.session_state.processing_time:.2f} seconds")

# Tab 2: Train Style Model
with tab2:
    st.markdown('<div class="main-header">Train Your Style Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Here you can train the humanizer to adapt to your preferred writing style.
        You can create new models or continue training existing ones with additional examples.
    </div>
    """, unsafe_allow_html=True)
    
    # Add option to create new model or continue training existing one
    training_mode = st.radio(
        "Training Mode:",
        ["Create New Model", "Continue Training Existing Model"],
        horizontal=True
    )
    
    if training_mode == "Continue Training Existing Model":
        # Get list of existing models
        model_files = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
        
        if not model_files:
            st.warning("No existing models found. Please create a new model first.")
            training_mode = "Create New Model"
        else:
            # Select model to continue training
            model_to_train = st.selectbox(
                "Select model to continue training:",
                options=model_files,
                help="Choose an existing model to continue training"
            )
            
            # Load the selected model first
            model_path = os.path.join(MODELS_DIR, f"{model_to_train}.pkl")
            if st.session_state.humanizer.load_style_model(model_path):
                st.success(f"Loaded model '{model_to_train}' for continued training")
            else:
                st.error(f"Failed to load model '{model_to_train}'")
                training_mode = "Create New Model"
    
    # Model name input with better description
    if training_mode == "Create New Model":
        st.markdown('<div class="sub-header">Name Your Style Model</div>', unsafe_allow_html=True)
        model_name = st.text_input(
            "Enter a unique name for your style model:", 
            "my_style",
            help="This name will be used to save and load your model"
        )
    else:
        # For continued training, use the selected model name
        model_name = model_to_train
        
        # Show current model stats if continuing training
        if hasattr(st.session_state.humanizer, 'style_trainer') and st.session_state.humanizer.style_trainer:
            trainer = st.session_state.humanizer.style_trainer
            st.info(f"""
                **Current model statistics:**
                - Vocabulary size: {len(trainer.vocab_set)} words
                - Average word length: {trainer.style_profile['avg_word_length']:.2f}
                - Average sentence length: {trainer.style_profile['avg_sentence_length']:.2f}
                - Vocabulary richness: {trainer.style_profile['vocabulary_richness']:.2f}
            """)
    
    # Sample text inputs
    st.markdown('<div class="sub-header">Sample Texts</div>', unsafe_allow_html=True)
    if training_mode == "Continue Training Existing Model":
        st.markdown("""
        Add additional samples to enhance your existing style model. The new samples 
        will be combined with your existing training data to refine the model.
        """)
    else:
        st.markdown("""
        Add at least 3 samples of the writing style you want to learn. Each sample should be at least a few sentences.
        The better your samples represent your style, the better the humanizer will adapt.
        """)
    
    # Create a container for samples
    samples_container = st.container()
    
    with samples_container:
        samples = []
        
        # First three required samples
        for i in range(1, 4):
            sample = st.text_area(
                f"Sample {i}:", 
                height=120, 
                key=f"sample_{i}",
                placeholder=f"Enter sample text in your preferred style... (sample {i})"
            )
            if sample.strip():
                samples.append(sample)
        
        # Add more samples button
        show_more = st.checkbox("Add more sample texts", value=False)
        
        # Show more samples if requested
        if show_more:
            num_extra_samples = st.number_input(
                "Number of additional samples:", 
                min_value=1, 
                max_value=10, 
                value=2
            )
            
            for i in range(4, 4 + num_extra_samples):
                sample = st.text_area(
                    f"Sample {i}:", 
                    height=120, 
                    key=f"sample_{i}",
                    placeholder=f"Enter additional sample text... (sample {i})"
                )
                if sample.strip():
                    samples.append(sample)
    
    # Train button with better styling
    train_col1, train_col2, train_col3 = st.columns([1, 2, 1])
    with train_col2:
        train_button = st.button("🧠 Train Model", use_container_width=True, type="primary")
    
    # Handle train button click
    if train_button:
        if len(samples) < 2:
            st.markdown("""
            <div class="warning-box">
                Please add at least two sample texts for effective style training.
            </div>
            """, unsafe_allow_html=True)
        elif training_mode == "Create New Model" and not model_name.strip():
            st.markdown("""
            <div class="warning-box">
                Please provide a name for your style model.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Modified train_model call for continuous training
            continue_training = (training_mode == "Continue Training Existing Model")
            
            with st.spinner("Training style model..."):
                # Train the humanizer on the provided texts with continue flag
                st.session_state.humanizer.train_on_texts(samples, continue_training=continue_training)
                
                # Save the model
                model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
                st.session_state.humanizer.save_style_model(model_path)
                st.session_state.loaded_model = model_name
            
            success_message = "refined and saved" if continue_training else "trained and saved"
            st.markdown(f"""
            <div class="success-box">
                <b>Success!</b> Your style model '{model_name}' has been {success_message}.
                You can now use it in the Humanize Text tab to transform text in your style.
            </div>
            """, unsafe_allow_html=True)
            
            # Add details about the trained model
            st.markdown("### Model Details")
            st.markdown(f"- **Model Name**: {model_name}")
            st.markdown(f"- **Number of New Samples**: {len(samples)}")
            st.markdown(f"- **Total Text Length Added**: {sum(len(s) for s in samples)} characters")
            
            # Add usage instructions
            st.info("Switch to the 'Humanize Text' tab and select your model from the dropdown to use it.")

# Tab 3: LLM Integration (previously Tab 4)
with tab3:
    st.markdown('<div class="main-header">LLM Integration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Connect to locally deployed Large Language Models to enhance the humanizer's capabilities.
        This allows you to leverage powerful AI models running on your own hardware.
    </div>
    """, unsafe_allow_html=True)
    
    # Add a debug mode toggle at the top
    debug_mode = st.checkbox("Enable Debug Mode", value=False, 
                           help="Show detailed connection information")
    
    if debug_mode:
        st.info("Debug mode enabled. Connection details will be shown in the terminal/console.")
    
    # Show options for different LLM providers
    st.markdown("### Available LLM Providers")
    
    # Display provider information in an expandable section
    from humanizer.llm.interface import SUPPORTED_PROVIDERS
    with st.expander("View supported LLM providers", expanded=False):
        for provider_id, provider_info in SUPPORTED_PROVIDERS.items():
            st.markdown(f"**{provider_id}**")
            st.markdown(f"- {provider_info['description']}")
            st.markdown(f"- API Type: {provider_info['api_type']}")
            if 'default_port' in provider_info:
                st.markdown(f"- Default Port: {provider_info['default_port']}")
            st.markdown(f"- More Info: [{provider_info['url']}]({provider_info['url']})")
            st.markdown("---")
    
    # Show available configurations
    st.markdown("### LLM Configurations")
    
    # Create tabs for "Use Existing" and "Create New"
    llm_tab1, llm_tab2 = st.tabs(["Use Existing Model", "Add New Model"])
    
    with llm_tab1:
        available_models = get_available_models()
        
        if available_models:
            # Select an existing model
            selected_model_idx = st.selectbox(
                "Select an LLM configuration:",
                options=range(len(available_models)),
                format_func=lambda i: available_models[i].get("name", f"Model {i}")
            )
            
            selected_model = available_models[selected_model_idx]
            
            # Show model details
            st.markdown("#### Model Details")
            st.markdown(f"**Name:** {selected_model.get('name')}")
            st.markdown(f"**Provider:** {selected_model.get('provider')}")
            
            if selected_model.get('api_url'):
                st.markdown(f"**API URL:** {selected_model.get('api_url')}")
            
            if selected_model.get('model_path'):
                st.markdown(f"**Model Path:** {selected_model.get('model_path')}")
            
            # Test connection and load model
            if st.button("Test Connection", key="test_conn_existing"):
                with st.spinner("Testing connection..."):
                    llm = LLMInterface(selected_model)
                    
                    # Set verbose mode if debug is enabled
                    if debug_mode:
                        st.write("Debug information:")
                        st.code(f"Provider: {selected_model.get('provider')}\n"
                               f"API URL: {selected_model.get('api_url')}\n"
                               f"Model: {selected_model.get('model_name', 'default')}")
                    
                    result = llm.test_connection()
                    
                    if result["success"]:
                        st.success(f"Connection successful: {result['message']}")
                        st.session_state.llm_interface = llm
                        st.session_state.llm_connected = True
                    else:
                        st.error(f"Connection failed: {result['message']}")
                        
                        # Show troubleshooting advice for Ollama
                        if selected_model.get('provider') == 'ollama':
                            st.markdown("""
                            ### Troubleshooting Ollama Connection:
                            1. Make sure Ollama is running: `ollama serve` in terminal
                            2. Check if your model is downloaded: `ollama list`
                            3. If not, download it: `ollama pull deepseek-r1:8b`
                            4. Verify model name is correct (case-sensitive)
                            5. Try a different model like "llama2" for testing
                            """)
                        
                        st.session_state.llm_connected = False
            
            # Option to use this LLM for generation
            if st.session_state.llm_connected and st.session_state.llm_interface:
                st.markdown("#### Test Generation")
                test_prompt = st.text_area(
                    "Enter a prompt to test the LLM:",
                    value="The quick brown fox jumps over the lazy",
                    height=100
                )
                
                test_params = st.expander("Advanced Parameters", expanded=False)
                with test_params:
                    max_tokens = st.slider("Max Tokens", min_value=10, max_value=1000, value=100)
                    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7)
                    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95)
                
                if st.button("Generate Text", key="generate_test"):
                    with st.spinner("Generating text..."):
                        try:
                            generated_text = st.session_state.llm_interface.generate_text(
                                test_prompt, 
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )
                            
                            if generated_text:
                                st.markdown("#### Generated Output")
                                st.markdown(f"**Prompt:** {test_prompt}")
                                st.markdown("**Completion:**")
                                st.text_area("", generated_text, height=200, disabled=True)
                            else:
                                st.warning("No text was generated. Check the LLM connection.")
                        except Exception as e:
                            st.error(f"Error generating text: {str(e)}")
        else:
            st.info("No LLM configurations found. Add a new configuration in the 'Add New Model' tab.")
    
    with llm_tab2:
        st.markdown("#### Add New LLM Configuration")
        
        # Form for adding a new LLM config
        with st.form("add_llm_form"):
            # Basic configuration
            model_name = st.text_input("Configuration Name", value="My LLM")
            provider = st.selectbox("LLM Provider", options=list(SUPPORTED_PROVIDERS.keys()))
            
            # Provider-specific configuration
            st.markdown("#### Provider Settings")
            
            if SUPPORTED_PROVIDERS[provider]["api_type"] == "http":
                default_port = SUPPORTED_PROVIDERS[provider].get("default_port", 8000)
                api_url = st.text_input(
                    "API URL",
                    value=f"http://localhost:{default_port}"
                )
                
                # Additional provider-specific fields
                if provider == "ollama":
                    model_identifier = st.text_input("Model Name", value="llama2")
                elif provider == "oobabooga":
                    model_identifier = st.text_input("Preset Name (optional)", value="")
                else:
                    model_identifier = ""
                
            elif provider == "gpt4all":
                api_url = ""
                model_identifier = st.text_input(
                    "Model Path",
                    value=str(Path.home() / ".cache" / "gpt4all" / "ggml-model.bin")
                )
            
            # Submit button
            submitted = st.form_submit_button("Add Configuration")
            
            if submitted:
                # Create the configuration
                config = {
                    "name": model_name,
                    "provider": provider,
                }
                
                if api_url:
                    config["api_url"] = api_url
                
                if model_identifier:
                    if provider == "ollama":
                        config["model_name"] = model_identifier
                    elif provider == "oobabooga":
                        config["preset"] = model_identifier
                    elif provider == "gpt4all":
                        config["model_path"] = model_identifier
                
                # Add the configuration
                llm = LLMInterface()
                if llm.add_model_config(config):
                    st.success(f"Added LLM configuration: {model_name}")
                else:
                    st.error("Failed to add LLM configuration")
    
    # Add a section explaining how to use LLM for humanization
    with st.expander("How to Use LLM for Humanization", expanded=True):
        st.markdown("""
        ### Using LLM for Text Humanization
        
        You can leverage the power of local LLMs to humanize your text with more advanced capabilities:
        
        1. **Connect to an LLM**: Configure and connect to a local LLM above
        2. **Go to the Humanize Text tab**: Once connected, you'll see a new option "Use LLM for Humanization"
        3. **Select a humanization style**: Choose from different prompts optimized for various use cases
        4. **Process your text**: The LLM will transform your text according to the selected style
        
        #### Available Styles:
        - **Standard**: General humanization for more natural text
        - **Avoid Detection**: Rewrite to bypass AI content detection
        - **Academic**: Simplify complex academic writing
        - **Casual**: Convert to a casual, conversational style
        - **Professional**: Create clear, concise professional content
        
        For best results, use models with at least 7B parameters.
        """)

# Add an about section in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    Text Humanizer uses natural language processing to transform text to be more human-like.
    It can help with:
    - Making AI-generated text appear more natural
    - Simplifying complex academic writing
    - Personalizing content to match your writing style
    """)
    
    st.markdown("## Tips")
    st.markdown("""
    - For best results, train with at least 3-5 samples of text
    - Higher style strength values will make more aggressive changes
    - After humanizing, manually review the text for best results
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Made with ❤️ by Text Humanizer | "
    "© 2023 Text Humanizer</div>", 
    unsafe_allow_html=True
)
