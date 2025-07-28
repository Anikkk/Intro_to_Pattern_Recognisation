# # app.py
# import streamlit as st
# from PIL import Image
# from backend import predict

# st.set_page_config(page_title="Veggie Classifier", layout="centered")
# st.title("🥦🥜🌶️🥭 Veggie & Nut Classifier")

# st.write("Upload an image of a **mango**, **pepper**, **nut**, or **broccoli**.")

# uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     if st.button("Predict"):
#         with st.spinner("Predicting..."):
#             label = predict(image)
#             st.success(f"🧠 Predicted Class: **{label}**")


import streamlit as st
from PIL import Image
from backend import classify_image

st.set_page_config(page_title="Veggie Classifier", layout="centered")
st.title("🥦🥜🌶️🥭 Veggie & Nut Classifier")
st.write("Upload an image of a **mango**, **pepper**, **nut**, or **broccoli** to classify its type and preparation state (chopped, sliced, or whole).")

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            produce_result, variation_result = classify_image(image)
            
            # Display produce classification
            st.write("### Produce Classification")
            st.success(f"🧠 Predicted Produce: **{produce_result['class']}** (Confidence: {produce_result['confidence']:.4f})")
            
            # Display variation classification
            st.write("### Preparation State")
            if variation_result['class'] != 'unknown':
                st.success(f"🧠 Predicted Preparation: **{variation_result['class']}** (Confidence: {variation_result['confidence']:.4f})")
            else:
                st.error(f"No variation classifier available for {produce_result['class']}.")