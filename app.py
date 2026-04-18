import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2

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

    if input_shape[1] == 3:
        height = input_shape[2]
        width = input_shape[3]
        channel_first = True
    else:
        height = input_shape[1]
        width = input_shape[2]
        channel_first = False

    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32) / 255.0

    if channel_first:
        image = np.transpose(image, (2,0,1))

    image = np.expand_dims(image, axis=0)

    return image


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict(image):

    input_data = preprocess(image, session)

    prediction = session.run(
        [output_name],
        {input_name: input_data}
    )[0]

    pred_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_letter = labels.get(pred_index, "Unknown")

    return predicted_letter, confidence


# -------------------------
# INPUT TYPE SELECTION
# -------------------------
input_type = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Camera"]
)


# ==========================================================
# IMAGE UPLOAD MODE
# ==========================================================
if input_type == "Upload Image":

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Predict Images"):

            for file in uploaded_files:

                img = Image.open(file).convert("RGB")
                img_np = np.array(img)

                pred, conf = predict(img_np)

                st.image(img, caption=file.name, width=250)
                st.success(f"Prediction: {pred}")
                st.info(f"Confidence: {conf:.4f}")

                st.divider()


# ==========================================================
# CAMERA MODE WITH BOUNDING BOX
# ==========================================================
else:

    camera_image = st.camera_input("Take a picture")

    if camera_image:

        img = Image.open(camera_image).convert("RGB")
        frame = np.array(img)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)

            if w > 20 and h > 20:

                roi = frame[y:y+h, x:x+w]

                pred, conf = predict(roi)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0,255,0),
                    3
                )

                cv2.putText(
                    frame,
                    f"{pred} ({conf:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                st.image(roi, caption="Detected Hand")

        st.image(frame, caption="Detection Result")
