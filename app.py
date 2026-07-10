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
        map_location="cpu"
    )

    classes = checkpoint["classes"]

    model = CNN(len(classes))

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model, classes



model, class_names = load_model()
