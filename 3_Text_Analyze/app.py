import gradio as gr
import google.generativeai as genai
import os
import textstat  # For readability and complexity score

# Ensure the API key is securely set
GOOGLE_API_KEY = os.getenv("API")  # Load API key from environment variable
genai.configure(api_key=GOOGLE_API_KEY)

# Load the Gemini Model
model = genai.GenerativeModel(model_name="models/gemini-2.0-pro")  # Upgraded model


def analyze_input(text, file):
    try:
        if file is not None:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
        elif not text.strip():
            return "⚠️ Error: Please enter text or upload a file.", "", "", ""

        text = text[:3000]  # Increased input text limit
        prompt = f"Analyze, summarize, and extract key insights from this document:\n\n{text}"
        response = model.generate_content([prompt], stream=False)
        result = response.text if response else "No response from AI."
        word_count = len(text.split())

        # Additional insights
        insight_prompt = f"Provide key topics, sentiment analysis, and readability score for this document:\n\n{text}"
        insight_response = model.generate_content([insight_prompt], stream=False)
        insights = insight_response.text if insight_response else "No insights available."

        # Readability score
        readability_score = textstat.flesch_reading_ease(text)
        grammar_check = f"Readability Score: {readability_score:.2f} (Higher is easier to read)"

        return result, f"📊 Word Count: {word_count}", insights, grammar_check
    except Exception as e:
        return f"⚠️ Error: {str(e)}", "", "", ""


def clear_inputs():
    return "", None, "", "", "", ""


def generate_downloadable_file(text):
    if text.strip():
        file_path = "analysis_result.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return file_path
    else:
        return None


# Gradio UI with an enhanced layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 **AI-Powered Text & File Analyzer**  
    🚀 Upload a `.txt` file or enter text manually to get an AI-generated analysis, summary, and insights.
    """)

    with gr.Row():
        text_input = gr.Textbox(label="✍️ Enter Text", placeholder="Type or paste your text here...", lines=6)
        file_input = gr.File(label="📂 Upload Text File (.txt)", type="filepath")

    with gr.Row():
        analyze_button = gr.Button("🔍 Analyze", variant="primary")
        clear_button = gr.Button("🗑️ Clear", variant="secondary")

    with gr.Column():
        output_text = gr.Textbox(label="📝 Analysis & Summary", lines=10, interactive=False)
        word_count_display = gr.Textbox(label="📊 Word Count", interactive=False)
        insights_display = gr.Textbox(label="🔎 AI Insights (Topics & Sentiment)", lines=5, interactive=False)
        readability_display = gr.Textbox(label="📖 Readability & Grammar Check", interactive=False)

    with gr.Row():
        download_button = gr.Button("⬇️ Download Result", variant="success", size="sm")
        download_file = gr.File(label="📄 Click to Download", interactive=False)

    analyze_button.click(analyze_input, inputs=[text_input, file_input], outputs=[output_text, word_count_display, insights_display, readability_display])
    clear_button.click(clear_inputs, inputs=[], outputs=[text_input, file_input, output_text, word_count_display, insights_display, readability_display, download_file])
    download_button.click(generate_downloadable_file, inputs=output_text, outputs=download_file)

# Launch the Gradio app
demo.launch()
