import streamlit as st
import torch
import torch.nn as nn
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
# CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color:#0e1117;
}

.title-container {
    text-align:center;
    padding:20px;
}

.title-container h1 {
    font-size:2.6rem;
    background:linear-gradient(
        90deg,
        #34d399,
        #22c55e
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.title-container p {
    color:#9ca3af;
}


div[data-testid="stFileUploader"] {
    border:2px dashed #22c55e;
    border-radius:14px;
}


div.stButton > button {

    background:#22c55e;
    color:white;
    font-weight:bold;
    width:100%;
    border-radius:10px;
}


.result-card {

    background:rgba(34,197,94,0.12);
    border:1px solid #22c55e;
    border-radius:15px;
    padding:20px;
    margin-top:20px;
}


.info-card {

    background:rgba(255,255,255,0.05);
    border-left:4px solid #22c55e;
    padding:15px;
    margin-top:15px;
    border-radius:10px;

}


.footer {

    text-align:center;
    margin-top:40px;
    color:#6b7280;

}

</style>
""", unsafe_allow_html=True)



# ==========================
# Title
# ==========================

st.markdown("""
<div class="title-container">

<h1>🌿 Multi Crop Disease Detection</h1>

<p>
Upload a leaf image to identify crop diseases
</p>

</div>
""",
unsafe_allow_html=True)



# ==========================
# PyTorch CNN Model
# ==========================

class CNN(nn.Module):

    def __init__(self, classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*14*14,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,classes)
        )


    def forward(self,x):

        x=self.features(x)
        x=self.classifier(x)

        return x



# ==========================
# Load PyTorch Model
# ==========================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "model.pth",
        map_location="cpu",
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = CNN(len(classes))

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model, classes



model, class_names = load_model()



# ==========================
# Disease Information
# ==========================

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "description":
        "Bacterial infection causing small dark spots on leaves and fruits.",
        "treatment":
        "Use copper-based bactericides. Remove infected leaves and avoid overhead watering."
    },

    "Pepper__bell___healthy": {
        "description":
        "The leaf appears healthy without visible disease symptoms.",
        "treatment":
        "No treatment needed. Maintain proper watering and nutrients."
    },

    "Potato___Early_blight": {
        "description":
        "Fungal disease causing brown spots with ring patterns on leaves.",
        "treatment":
        "Use recommended fungicides and remove infected plant parts."
    },

    "Potato___Late_blight": {
        "description":
        "Serious fungal infection causing dark water-soaked lesions.",
        "treatment":
        "Apply fungicides and remove infected plants quickly."
    },

    "Potato___healthy": {
        "description":
        "The leaf appears healthy.",
        "treatment":
        "Continue normal crop maintenance."
    },

    "Tomato_Bacterial_spot": {
        "description":
        "Bacterial disease causing small dark spots on tomato leaves.",
        "treatment":
        "Use copper sprays and remove infected material."
    },

    "Tomato_Early_blight": {
        "description":
        "Fungal disease with target-like brown spots.",
        "treatment":
        "Apply fungicides and improve air circulation."
    },

    "Tomato_Late_blight": {
        "description":
        "Fast spreading fungal disease causing dark lesions.",
        "treatment":
        "Apply fungicides and destroy infected leaves."
    },

    "Tomato_Leaf_Mold": {
        "description":
        "Fungal disease producing yellow patches and mold growth.",
        "treatment":
        "Reduce humidity and improve ventilation."
    },

    "Tomato_Septoria_leaf_spot": {
        "description":
        "Small circular spots with dark borders.",
        "treatment":
        "Remove infected leaves and apply fungicide."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description":
        "Tiny pests causing yellowing and leaf damage.",
        "treatment":
        "Use insecticidal soap or neem oil."
    },

    "Tomato__Target_Spot": {
        "description":
        "Disease causing circular target-like spots.",
        "treatment":
        "Use fungicides and remove infected debris."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description":
        "Virus causing leaf curling and yellowing.",
        "treatment":
        "Control whiteflies and remove infected plants."
    },

    "Tomato__Tomato_mosaic_virus": {
        "description":
        "Virus causing mosaic patterns and leaf distortion.",
        "treatment":
        "Remove infected plants and control insects."
    },

    "Tomato_healthy": {
        "description":
        "The leaf appears healthy.",
        "treatment":
        "No treatment required."
    }
}



# ==========================
# Image Preprocessing
# ==========================

def preprocess_image(image):

    img = image.resize(
        (128, 128)
    )

    img = np.array(img)

    img = img / 255.0


    # HWC -> CHW

    img = np.transpose(
        img,
        (2, 0, 1)
    )


    img = torch.tensor(
        img,
        dtype=torch.float32
    )


    img = img.unsqueeze(0)

    return img



# ==========================
# Leaf Check
# ==========================

def is_leaf_like(image):

    img = np.array(
        image.resize((128, 128))
    )


    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]


    green = ((g > r) & (g > b))


    ratio = np.mean(green)


    return ratio > 0.15, ratio



def entropy_calc(pred):

    p = pred[0]

    return -np.sum(
        p * np.log(p + 1e-9)
    )



# ==========================
# Upload Image
# ==========================

uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )



    if st.button("🔍 Predict Disease"):


        leaf_ok, green_ratio = is_leaf_like(
            image
        )


        img_tensor = preprocess_image(
            image
        )


        with torch.no_grad():


            output = model(
                img_tensor
            )


            prob = torch.softmax(
                output,
                dim=1
            )


            prediction = prob.numpy()



        index = np.argmax(
            prediction
        )


        confidence = np.max(
            prediction
        ) * 100


        entropy = entropy_calc(
            prediction
        )



        with st.expander("🔧 Debug Info"):

            st.write(
                "Green ratio:",
                round(green_ratio, 4)
            )

            st.write(
                "Class index:",
                index
            )

            st.write(
                "Prediction:",
                class_names[index]
            )

            st.write(
                "Confidence:",
                round(confidence, 2),
                "%"
            )



        if not leaf_ok:

            st.error(
                "❌ Please upload a clear leaf image"
            )


        elif confidence < 75 or entropy > 1.2:

            st.error(
                f"❌ Low confidence image\n"
                f"Confidence: {confidence:.2f}%"
            )


        else:


            disease = class_names[index]

            info = disease_info.get(
                disease,
                {}
            )


            st.markdown(
                f"""
                <div class="result-card">

                <h3>🌿 {disease}</h3>

                <p>
                Confidence:
                <b>{confidence:.2f}%</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.progress(
                int(confidence)
            )


            st.markdown(
                f"""
                <div class="info-card">

                <h4>📖 Description</h4>

                <p>
                {info.get("description", "")}
                </p>

                </div>


                <div class="info-card">

                <h4>💊 Treatment</h4>

                <p>
                {info.get("treatment", "")}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )



# ==========================
# Footer
# ==========================

st.markdown(
"""
<div class="footer">

Developed by <b>Thamaraiselvan</b>

</div>
""",
unsafe_allow_html=True
)
