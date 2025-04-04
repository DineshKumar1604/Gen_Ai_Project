import gradio as gr
from huggingface_hub import InferenceClient, auth_check
from deep_translator import GoogleTranslator
from PIL import Image
from gradio.themes import Base
import os
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# Fetch the API token from environment variable
hf_api_token = os.getenv("HF_API_TOKEN")

# Hugging Face Inference API client
client = InferenceClient(token=hf_api_token)

# Supported languages for translation (aligned with deep_translator)
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
}

# Check token access to the model
def check_model_access():
    try:
        auth_check("Salesforce/blip-image-captioning-large", token=hf_api_token)
        return "Token has access to the model."
    except GatedRepoError:
        return "Error: Token does not have permission to access this gated repository."
    except RepositoryNotFoundError:
        return "Error: The repository was not found or you do not have access."
    except Exception as e:
        return f"Error checking access: {str(e)}"

# Print access check result (for debugging)
print(check_model_access())

def generate_caption(image, target_language_name):
    try:
        # Map the selected language name to its code
        target_language = languages.get(target_language_name)
        if not target_language:
            return f"Error: Selected language '{target_language_name}' is not supported. Please choose from: {list(languages.keys())}"

        # Convert PIL image to bytes for API
        from io import BytesIO
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Use Hugging Face Inference API for captioning
        result = client.image_to_text(
            image=img_byte_arr,
            model="Salesforce/blip-image-captioning-large"
        )

        # Extract the generated text from the ImageToTextOutput object
        english_caption = result.generated_text

        # If target language is English, return as is
        if target_language == "en":
            return english_caption
        
        # Translate to the selected local language
        translator = GoogleTranslator(source='en', target=target_language)
        local_caption = translator.translate(english_caption)
        
        return local_caption
    
    except Exception as e:
        return f"Error: {str(e)}"

# Custom theme
custom_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    text_size="lg",
    radius_size="md",
    font=[gr.themes.GoogleFont("Roboto"), "sans-serif"]
)

# Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Dropdown(
            choices=list(languages.keys()),
            label="Select Language",
            value="English"
        )
    ],
    outputs=gr.Textbox(label="Caption", lines=2, placeholder="Caption will appear here..."),
    title="Image Caption Generator with Language Selection",
    description="Upload an image and select a local language to get a caption.",
    theme=custom_theme,
    css="""
        .gradio-container { max-width: 800px; margin: auto; }
        h1 { text-align: center; color: #1E40AF; }
        .label { font-weight: bold; }
        input, output { border-radius: 8px; }
    """
)

interface.launch()
