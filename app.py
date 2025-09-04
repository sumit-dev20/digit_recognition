import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Digit Recognition", page_icon="🔢")

st.title("🔢 Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and let the model recognize it!")

model = load_model('model.keras')

# -------------------------------
# Drawing Canvas
# -------------------------------
st.write("Draw your digit below 👇")

canvas_result = st_canvas(
    fill_color="red",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------------
# Prediction
# -------------------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # A blank canvas will have very low pixel intensity (all black)
    if np.mean(gray) < 5:   # threshold for "empty" canvas
        st.info("✏️ Please draw a digit above to get prediction.")
    else:
        gray_resized = cv2.resize(gray, (28, 28))
        gray_resized = gray_resized.astype("float32") / 255.0
        gray_resized = np.expand_dims(gray_resized, axis=(0, -1))

        prediction = model.predict(gray_resized)
        predicted_class = np.argmax(prediction)

        st.subheader(f"Prediction: **{predicted_class}**")
        st.bar_chart(prediction[0])