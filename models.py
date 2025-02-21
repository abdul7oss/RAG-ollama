@st.cache_resource(show_spinner=True)
def extract_model_names() -> Tuple[str, ...]:
    """Extracts available model names from Ollama API with proper structure handling."""
    logger.info("Extracting model names")
    try:
        models_info = ollama.list()

        # Ensure response is valid
        if not models_info or not hasattr(models_info, "models") or not models_info.models:
            logger.error("No models found in Ollama API response.")
            return ()
        
        # Extract model names properly
        model_names = tuple(
            model.model for model in models_info.models if model.model and "llama2" not in model.model
        )

        if not model_names:
            logger.error("No valid models found in response.")
            return ()

        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return ()