import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.title("FSL ABC Image Classifier")

@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess(image):
    image = image.resize((64,64))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


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

        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {conf:.4f}")
