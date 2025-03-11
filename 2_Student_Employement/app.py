import gradio as gr
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_employability(*features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return "Congrats, You're Employable!" if prediction[0] == 1 else "Better luck next time, You're Less Employable"

with gr.Blocks() as demo:
    gr.Markdown("# Student Employability Prediction")
    gr.Markdown(
        "Enter your ratings for the given parameters (1-5) using the sliders below. "
        "Click 'Submit' to check if the student is **Employable** or **Less Employable**."
    )

    with gr.Row():
        with gr.Column():
            sliders = [
                gr.Slider(1, 5, value=3, step=1, label="General Appearance"),
                gr.Slider(1, 5, value=3, step=1, label="Manner of Speaking"),
                gr.Slider(1, 5, value=3, step=1, label="Physical Condition"),
                gr.Slider(1, 5, value=3, step=1, label="Mental Alertness"),
                gr.Slider(1, 5, value=3, step=1, label="Self-Confidence"),
                gr.Slider(1, 5, value=3, step=1, label="Ability to Present Ideas"),
                gr.Slider(1, 5, value=3, step=1, label="Communication Skills"),
                gr.Slider(1, 5, value=3, step=1, label="Student Performance Rating"),
            ]
        with gr.Column():
            output = gr.Textbox(label="Employability Status")

    submit = gr.Button("Submit")
    submit.click(predict_employability, inputs=sliders, outputs=output)

demo.launch()
