import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("FSL ABC Image Classifier")

# Class mapping
labels = {
  0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
  5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
  10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
  15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
  20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
  25: 'Z'
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
