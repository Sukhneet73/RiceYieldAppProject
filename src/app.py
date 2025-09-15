import pandas as pd
import numpy as np
import pickle
import json
import gradio as gr
import shap
import os
from dotenv import load_dotenv

# Load OpenAI key (optional)
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
use_openai = bool(OPENAI_KEY)

try:
    import openai
    if use_openai:
        openai.api_key = OPENAI_KEY
except:
    use_openai = False

# Load trained model (updated paths)
with open("../rice_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../feature_names.json", "r") as f:
    feature_names = json.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Function for farmer-friendly explanations
def farmer_friendly_explanation(feature_imp):
    notes = []
    tips = []

    for feature, val in feature_imp.items():
        if feature == "rainfall":
            notes.append(f"🌧️ Rainfall: {'Good' if val > 0 else 'Low, may reduce yield'}")
            if val < 0:
                tips.append("💡 Consider irrigation to improve water supply.")
        elif feature == "temperature":
            notes.append(f"🌡️ Temperature: {'Ideal for rice' if val > 0 else 'Not ideal for rice'}")
            if val < 0:
                tips.append("💡 Avoid planting during extreme temperatures.")
        elif feature == "soil_type":
            notes.append(f"🌱 Soil type: {'Suitable for rice' if val > 0 else 'Not very suitable'}")
            if val < 0:
                tips.append("💡 Use soil amendments or fertilizers to improve soil quality.")
        elif feature == "fertilizer":
            notes.append(f"💧 Fertilizer: {'Helps yield' if val > 0 else 'Less fertilizer may reduce yield'}")
            if val < 0:
                tips.append("💡 Apply recommended fertilizer doses for better yield")

    # Add two newlines after each note and tip to force Markdown line breaks
    notes_text = "\n\n".join(notes)
    tips_text = "\n\n".join(tips)
    return f"{notes_text}\n\n**Tips:**\n{tips_text}"

# Prediction function
def predict(rainfall, temperature, soil_type, fertilizer):
    df = pd.DataFrame([[rainfall, temperature, soil_type, fertilizer]], columns=feature_names)
    pred = model.predict(df)[0]
    shap_values = explainer.shap_values(df)
    
    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values)[0][0]
    else:
        shap_arr = shap_values[0] if shap_values.shape[0] == 1 else shap_values[0]
    
    feature_imp = {name: float(round(float(v), 2)) for name, v in zip(feature_names, shap_arr)}
    
    note = farmer_friendly_explanation(feature_imp)
    
    return round(float(pred), 2), note

# Gradio UI
inputs = [
    gr.Number(label="Rainfall (mm)", value=180),
    gr.Number(label="Temperature (°C)", value=29),
    gr.Number(label="Soil Type (1 = sandy, 2 = clay)", value=1),
    gr.Number(label="Fertilizer (kg)", value=55),
]

outputs = [
    gr.Number(label="Predicted Yield"),
    gr.Markdown(label="Farmer-friendly explanation")  # Multi-line support
]

title = "🌾 Rice Yield Predictor"
desc = "Enter field values. The app predicts yield and shows interactive tips for farmers."

iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc,allow_flagging="never")

if __name__ == "__main__":
    iface.launch()
