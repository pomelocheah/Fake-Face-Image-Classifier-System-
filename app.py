import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision import models
from model import build_cnn, build_mobilenet

# -------------------------- Model Loading Logic --------------------------


import tensorflow as tf
from PIL import Image
import numpy as np

def load_model():
    # Load keras model instead of torch.load
    model = tf.keras.models.load_model("models/B_MobileNet_baseline.keras")
    return model

# Global load once
model = load_model()

def predict_image(input_img: Image.Image):
    if input_img is None:
        return "Please upload a face image", None

    img = input_img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    # MobileNet专用预处理，替换 /255
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)
    

    img_batch = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_batch, verbose=0)[0]
    print("Model raw output value:", pred[0])

    real_conf = float(pred[0])
    fake_conf = 1.0 - real_conf

    if real_conf > fake_conf:
        label = "✅ Real Human Face"
    else:
        label = "❌ AI-Generated Fake Face"

    result_md = f"""
### Detection Result
{label}

Confidence Scores:
- Real Face: {real_conf:.2%}
- Fake Face: {fake_conf:.2%}
"""
    return result_md, input_img

# --------------------------  UI Layout --------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(theme=theme, title="Real vs Fake Face Detector") as demo:
    gr.Markdown("""
    # 🕵️ Real / Fake Face Detector
    Upload a portrait photo, and AI will identify whether it is a real human face or AI-generated fake face.
    """)

    with gr.Row(variant="panel"):
        # Left Column: Upload Area
        with gr.Column(scale=1):
            upload_image = gr.Image(
                type="pil",
                label="Upload Face Image",
                height=400
            )
            submit_button = gr.Button("Start Detection", variant="primary", size="lg")
            clear_button = gr.Button("Clear All", variant="secondary")

        # Right Column: Preview & Result
        with gr.Column(scale=1):
            preview_image = gr.Image(label="Image Preview", height=400)
            output_result = gr.Markdown()

    # Button trigger binding
    submit_button.click(
        fn=predict_image,
        inputs=[upload_image],
        outputs=[output_result, preview_image]
    )
    clear_button.click(
        lambda: (None, "", None),
        outputs=[upload_image, output_result, preview_image]
    )

if __name__ == "__main__":
    # Config for cloud deployment
  demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme, share=True)