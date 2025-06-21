import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Generator architecture (must match train_cgan.py)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 10
        latent_dim = 100
        embedding_dim = 50
        img_shape = (1, 28, 28)
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        input_dim = latent_dim + embedding_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        img = self.model(gen_input)
        return img.view(img.size(0), *self.img_shape)

@st.cache_resource
def load_generator(path="generator.pt"):
    device = torch.device("cpu")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(path, map_location=device))
    gen.eval()
    return gen

generator = load_generator()

st.title("🖋️ Handwritten Digit Generator")
selected_digit = st.slider("Select a digit to generate", 0, 9, 0)

if st.button("Generate 5 Samples"):
    noise = torch.randn(5, 100)
    labels = torch.full((5,), selected_digit, dtype=torch.long)
    with torch.no_grad():
        samples = generator(noise, labels).cpu()
    samples = ((samples + 1) * 127.5).clamp(0, 255).byte()
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        img_arr = samples[idx].squeeze().numpy()
        img = Image.fromarray(img_arr, mode="L")
        col.image(img, caption=f"Digit {selected_digit}", use_column_width=True)

# requirements.txt
# torch
# torchvision
# streamlit
# numpy
# pillow