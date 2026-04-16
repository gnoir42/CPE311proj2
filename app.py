import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("FSL ABC Image Classifier")

# Class mapping
labels = {
    0: "A",
    1: "B",
    2: "C"
}

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):

        processed = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Convert index to letter
        predicted_letter = labels.get(pred_class, "Unknown")

        st.success(f"Prediction: {predicted_letter}")
        st.info(f"Confidence: {confidence:.4f}")
