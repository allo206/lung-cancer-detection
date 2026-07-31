import streamlit as st
from PIL import Image
from predict import predict_lung_image

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏥 MedScan AI")
st.subheader("Lung Cancer Detection System")
st.write("Upload a Lung CT Scan image for AI-assisted analysis.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a CT Scan Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# If Image Uploaded
# -----------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded CT Scan",
        use_container_width=True
    )

    img.save("temp_image.jpg")

    st.write("")

    if st.button("🔍 Analyze Scan"):

        with st.spinner("Analyzing CT Scan..."):

            predicted_class, confidence = predict_lung_image("temp_image.jpg")

        confidence = float(confidence)

        st.divider()

        st.subheader(" Analysis Report")

        if predicted_class == "Normal Lung":

            st.success("🟢 Normal Lung")

        elif predicted_class == "Adenocarcinoma":

            st.error("🔴 Adenocarcinoma")

        else:

            st.error("🔴 Squamous Cell Carcinoma")

        st.write(f"### Confidence : {confidence:.2f}%")

        st.progress(float(confidence) / 100)

        st.divider()

        st.subheader("Recommendation")

        if predicted_class == "Normal Lung":

            st.success("No signs of lung cancer detected.")

        else:

            st.warning(
                "Please consult a qualified medical specialist for further diagnosis."
            )