import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.title("FSL ABC Image Classifier")

# Class label mapping
labels = {
  0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
  10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
  19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# -----------------------------
# MODEL SELECTION
# -----------------------------
model_choice = st.selectbox(
    "Choose Model",
    ("Baseline CNN", "Hypertuned CNN", "MobileNet")
)

model_files = {
    "Baseline CNN": "baseline_quantized.onnx",
    "Hypertuned CNN": "model.onnx",
    "MobileNet": "mobilenet.onnx"
}

model_path = model_files[model_choice]


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model(model_path):
    return ort.InferenceSession(model_path)

session = load_model(model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess(image):
    image = image.resize((64,64))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img)

    if st.button("Predict"):

        input_data = preprocess(img)

        prediction = session.run(
            [output_name],
            {input_name: input_data}
        )[0]

        pred = int(np.argmax(prediction))
        conf = float(np.max(prediction))

        predicted_letter = labels.get(pred, "Unknown")

        st.success(f"Prediction: {predicted_letter}")
        st.info(f"Confidence: {conf:.4f}")
