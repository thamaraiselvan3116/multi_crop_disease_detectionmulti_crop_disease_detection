import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================
# Page Setup
# ==========================
st.set_page_config(
    page_title="Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# ==========================
# Custom CSS Styling
# ==========================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }

    .title-container {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .title-container h1 {
        font-size: 2.6rem;
        background: linear-gradient(90deg, #34d399, #22c55e, #16a34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }

    .title-container p {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 4px;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #22c55e;
        border-radius: 14px;
        padding: 18px;
        background-color: rgba(34, 197, 94, 0.06);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 10px 26px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.25s;
        box-shadow: 0 4px 14px rgba(34, 197, 94, 0.35);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(34, 197, 94, 0.5);
    }

    .result-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(22,163,74,0.06));
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 14px;
    }

    .result-card h3 {
        color: #4ade80;
        margin-bottom: 6px;
    }

    .info-card {
        background: rgba(255,255,255,0.03);
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 12px;
    }

    .info-card h4 {
        color: #e5e7eb;
        margin-bottom: 6px;
        font-size: 1.05rem;
    }

    .info-card p {
        color: #cbd5e1;
        margin: 0;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        padding: 28px 0 10px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #6b7280;
        font-size: 0.9rem;
    }

    .footer span {
        color: #4ade80;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================
# Title
# ==========================
st.markdown("""
    <div class="title-container">
        <h1>🌿 Multi Crop Disease Detection</h1>
        <p>Upload a leaf image to identify crop diseases instantly</p>
    </div>
""", unsafe_allow_html=True)


# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model


model = load_model()


# ==========================
# Class Names
# ==========================
class_names = [
    "Pepper Bell - Bacterial Spot",       # 0
    "Pepper Bell - Healthy",              # 1
    "Potato - Early Blight",              # 2
    "Potato - Late Blight",               # 3
    "Potato - Healthy",                   # 4
    "Tomato - Bacterial Spot",            # 5
    "Tomato - Early Blight",              # 6
    "Tomato - Late Blight",               # 7
    "Tomato - Leaf Mold",                 # 8
    "Tomato - Septoria Leaf Spot",        # 9
    "Tomato - Spider Mites",              # 10
    "Tomato - Target Spot",               # 11
    "Tomato - Yellow Leaf Curl Virus",    # 12
    "Tomato - Mosaic Virus",              # 13
    "Tomato - Healthy",                   # 14
]


# ==========================
# Disease Info: Description + Treatment
# ==========================
disease_info = {
    "Pepper Bell - Bacterial Spot": {
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves and fruit, which later turn brown and scabby.",
        "treatment": "Use copper-based bactericides. Avoid overhead watering. Remove and destroy infected plant debris. Rotate crops every season."
    },
    "Pepper Bell - Healthy": {
        "description": "The leaf shows no visible signs of disease or infection.",
        "treatment": "No treatment needed. Continue regular watering, fertilization, and monitoring."
    },
    "Potato - Early Blight": {
        "description": "Fungal disease causing dark brown spots with concentric rings, mostly on older leaves.",
        "treatment": "Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves. Practice crop rotation and avoid overhead irrigation."
    },
    "Potato - Late Blight": {
        "description": "Serious fungal disease causing dark, water-soaked lesions that spread rapidly in humid conditions.",
        "treatment": "Apply fungicides containing metalaxyl or chlorothalonil immediately. Destroy infected plants. Ensure good field drainage and air circulation."
    },
    "Potato - Healthy": {
        "description": "The leaf shows no visible signs of disease or infection.",
        "treatment": "No treatment needed. Continue regular watering, fertilization, and monitoring."
    },
    "Tomato - Bacterial Spot": {
        "description": "Bacterial disease causing small, dark, greasy-looking spots on leaves and fruit.",
        "treatment": "Apply copper-based sprays. Avoid working with wet plants. Use disease-free seeds and remove infected plant material."
    },
    "Tomato - Early Blight": {
        "description": "Fungal disease causing brown spots with target-like rings, usually starting on lower leaves.",
        "treatment": "Apply fungicides such as mancozeb or chlorothalonil. Remove affected leaves. Mulch soil and avoid overhead watering."
    },
    "Tomato - Late Blight": {
        "description": "Aggressive fungal disease causing large, irregular, water-soaked lesions that turn brown/black quickly.",
        "treatment": "Apply fungicides immediately (metalaxyl, chlorothalonil). Remove and destroy infected plants. Avoid excess moisture and improve air flow."
    },
    "Tomato - Leaf Mold": {
        "description": "Fungal disease causing pale yellow spots on top of leaves with olive-green mold underneath.",
        "treatment": "Improve ventilation, reduce humidity. Apply fungicides like chlorothalonil. Avoid overhead watering."
    },
    "Tomato - Septoria Leaf Spot": {
        "description": "Fungal disease causing small circular spots with dark borders and gray centers on lower leaves.",
        "treatment": "Remove infected leaves. Apply fungicide (chlorothalonil/mancozeb). Avoid overhead irrigation and rotate crops."
    },
    "Tomato - Spider Mites": {
        "description": "Pest damage caused by tiny mites feeding on leaves, causing stippling, yellowing, and fine webbing.",
        "treatment": "Spray with insecticidal soap or neem oil. Increase humidity around plants. Introduce natural predators like ladybugs."
    },
    "Tomato - Target Spot": {
        "description": "Fungal disease causing brown spots with concentric rings, resembling a target, on leaves and fruit.",
        "treatment": "Apply fungicides (chlorothalonil, azoxystrobin). Remove infected debris. Ensure proper plant spacing for airflow."
    },
    "Tomato - Yellow Leaf Curl Virus": {
        "description": "Viral disease spread by whiteflies, causing upward curling, yellowing, and stunted growth of leaves.",
        "treatment": "No cure available. Remove and destroy infected plants. Control whiteflies using insecticides or sticky traps. Use resistant varieties."
    },
    "Tomato - Mosaic Virus": {
        "description": "Viral disease causing mottled light/dark green patterns, leaf curling, and stunted plant growth.",
        "treatment": "No cure available. Remove infected plants immediately. Disinfect tools. Control aphids and avoid tobacco contact with plants."
    },
    "Tomato - Healthy": {
        "description": "The leaf shows no visible signs of disease or infection.",
        "treatment": "No treatment needed. Continue regular watering, fertilization, and monitoring."
    }
}


# ==========================
# Helper: Check if image looks like a leaf
# ==========================
def is_leaf_like(image, green_threshold=0.15):
    img_array = np.array(image.resize((128, 128)))
    r = img_array[:, :, 0].astype(int)
    g = img_array[:, :, 1].astype(int)
    b = img_array[:, :, 2].astype(int)
    green_mask = (g > r) & (g > b)
    green_ratio = np.mean(green_mask)
    return green_ratio > green_threshold, green_ratio


def prediction_entropy(prediction):
    p = prediction[0]
    entropy = -np.sum(p * np.log(p + 1e-9))
    return entropy


# ==========================
# Upload Image
# ==========================
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict Disease"):

        leaf_ok, green_ratio = is_leaf_like(image)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        entropy = prediction_entropy(prediction)

        with st.expander("🔧 Debug Info"):
            st.write("Green ratio (leaf check):", round(green_ratio, 4), "| Passed leaf check:", leaf_ok)
            st.write("Raw prediction values:", prediction)
            st.write("Predicted class index:", predicted_class)
            st.write("Predicted class name:", class_names[predicted_class])
            st.write("Confidence:", round(confidence, 2), "%")
            st.write("Entropy:", round(entropy, 4))

        ENTROPY_THRESHOLD = 1.2

        if not leaf_ok:
            st.error(
                f"❌ Unmatched Image!\n\nThis doesn't look like a crop leaf image "
                f"(green_ratio={round(green_ratio,3)}, needs > 0.15). "
                f"Please upload a clear leaf photo."
            )
        elif confidence < 75 or entropy > ENTROPY_THRESHOLD:
            st.error(
                f"❌ Unmatched Image!\n\nModel is not confident this is a valid "
                f"crop leaf (confidence={round(confidence,2)}%, entropy={round(entropy,3)}). "
                f"Please upload a clear crop leaf image."
            )
        else:
            disease = class_names[predicted_class]
            info = disease_info.get(disease)

            st.markdown(f"""
                <div class="result-card">
                    <h3>🌿 {disease}</h3>
                    <p style="color:#cbd5e1; margin-top:6px;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence))

            if info:
                st.markdown(f"""
                    <div class="info-card">
                        <h4>📖 Description</h4>
                        <p>{info['description']}</p>
                    </div>
                    <div class="info-card">
                        <h4>💊 Treatment</h4>
                        <p>{info['treatment']}</p>
                    </div>
                """, unsafe_allow_html=True)


# ==========================
# Footer
# ==========================
st.markdown("""
    <div class="footer">
        Developed by <span>Thamaraiselvan</span>
    </div>
""", unsafe_allow_html=True)
