import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.title("FSL ABC Image Classifier")

# -------------------------
# CLASS LABELS
# -------------------------
labels = {
0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',
19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# -------------------------
# MODEL SELECTION
# -------------------------
model_choice = st.selectbox(
    "Select Model",
    ["Baseline CNN", "Hypertuned CNN", "MobileNet"]
)

model_files = {
    "Baseline CNN": "baseline_quantized.onnx",
    "Hypertuned CNN": "model.onnx",
    "MobileNet": "mobilenet.onnx"
}

model_path = model_files[model_choice]


# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path)

session = load_model(model_path)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess(image, session):

    input_shape = session.get_inputs()[0].shape

    # detect expected height/width
    if input_shape[1] == 3:
        height = input_shape[2]
        width = input_shape[3]
        channel_first = True
    else:
        height = input_shape[1]
        width = input_shape[2]
        channel_first = False

    image = image.resize((width, height))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.float32) / 255.0

    if channel_first:
        image = np.transpose(image, (2,0,1))

    image = np.expand_dims(image, axis=0)

    return image


# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):

        input_data = preprocess(img, session)

        prediction = session.run(
            [output_name],
            {input_name: input_data}
        )[0]

        pred_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        predicted_letter = labels.get(pred_index, "Unknown")

        st.success(f"Prediction: {predicted_letter}")
        st.info(f"Confidence: {confidence:.4f}")
