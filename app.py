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
st.write("Upload a Lung Histopathology Image for AI-assisted Analysis.")

# -----------------------------
# Information
# -----------------------------
st.info("""
📌 Supported Image Type

✔ Lung Histopathology (Microscopic Tissue) Images Only
""")

st.warning("""
📌 Supported Tissue Types

🟢 Normal Lung Tissue

🔴 Lung Adenocarcinoma

🔴 Lung Squamous Cell Carcinoma

⚠ Important Warning

Please upload only supported lung histopathology images.

Images outside the supported tissue types may not be classified accurately.""")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a Histopathology Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# If Image Uploaded
# -----------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Histopathology Image",
        use_container_width=True
    )

    img.save("temp_image.jpg")

    st.write("")

    if st.button("🔍 Analyze Image"):

        with st.spinner("Analyzing Histopathology Image..."):

            predicted_class, confidence = predict_lung_image("temp_image.jpg")

        confidence = float(confidence)

        st.divider()

        st.subheader("Analysis Report")

        if predicted_class == "Normal Lung":

            st.success("🟢 Normal Lung Tissue")

        elif predicted_class == "Adenocarcinoma":

            st.error("🔴 Lung Adenocarcinoma")

        else:

            st.error("🔴 Lung Squamous Cell Carcinoma")

        st.write(f"### Confidence : {confidence:.2f}%")

        st.progress(confidence / 100)

        st.divider()

        st.subheader("Recommendation")

        if predicted_class == "Normal Lung":

            st.success("No signs of lung cancer detected in the uploaded histopathology image.")

        else:

            st.warning(
                "Please consult a qualified pathologist or oncologist for further diagnosis."
            )