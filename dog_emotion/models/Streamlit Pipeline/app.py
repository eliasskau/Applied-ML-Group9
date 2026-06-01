import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(
    page_title="Dog Emotion Detector",
    page_icon="🐶",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .stHeadingContainer h1 { color: #1E293B; font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Model & Weights Configuration
# ==========================================
NUM_CLASSES = 4
MODEL_WEIGHTS_PATH = "best_distilled_student.pth"

CLASS_MAPPING = {
    0: {"name": "Angry 🤬", "color": "#EF4444"},
    1: {"name": "Happy 😊", "color": "#10B981"},
    2: {"name": "Relaxed 😌", "color": "#3B82F6"},
    3: {"name": "Sad 😢", "color": "#F59E0B"}
}

@st.cache_resource
def load_distilled_model(weights_path):
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_distilled_model(MODEL_WEIGHTS_PATH)

# Image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Occlusion Sensitivity Core Logic
# ==========================================
def compute_occlusion_map(model, image, target_class, patch_size=32, stride=12):
    """
    Sliding-window occlusion to find which areas dramatically drop model confidence.
    """
    # Resize image to model input shape for mapping
    img_resized = image.resize((224, 224))
    img_tensor = preprocess(img_resized).unsqueeze(0) # [1, 3, 224, 224]
    
    # Get baseline probability without any occlusion
    with torch.no_grad():
        baseline_logits = model(img_tensor)
        baseline_prob = F.softmax(baseline_logits, dim=1)[0, target_class].item()
        
    _, _, height, width = img_tensor.shape
    heatmap = np.zeros((height, width))
    
    # Create gray pixel values to fill occluded patches (matching ImageNet normalization defaults)
    gray_patch = torch.zeros((3, patch_size, patch_size))
    gray_patch[0, :, :] = (0.485)
    gray_patch[1, :, :] = (0.456)
    gray_patch[2, :, :] = (0.406)

    # Slide window across the image dimensions
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Duplicate original tensor and apply patch mask
            occluded_tensor = img_tensor.clone()
            occluded_tensor[0, :, y:y+patch_size, x:x+patch_size] = gray_patch
            
            with torch.no_grad():
                logits = model(occluded_tensor)
                prob = F.softmax(logits, dim=1)[0, target_class].item()
            
            # Sensitivity = baseline confidence minus current confidence
            # Higher values mean that covering this spot hurt the model's confidence a lot!
            heatmap[y:y+patch_size, x:x+patch_size] += (baseline_prob - prob)

    # Post-process heatmap safely
    heatmap = np.maximum(heatmap, 0) # Keep positive importance shifts
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max() # Normalize to [0, 1] range
        
    return img_resized, heatmap

# ==========================================
# 4. Streamlit UI Layout
# ==========================================
st.title("🐶 Dog Emotion Recognition")
st.markdown("Upload a photo of a dog to analyze expressions via our distilled edge architecture.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Model Evaluation")
        with st.spinner("Analyzing canine expressions..."):
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = F.softmax(logits, dim=1)[0]
                
            top_prob, top_idx = torch.max(probabilities, dim=0)
            top_idx = top_idx.item()
            predicted_class = CLASS_MAPPING[top_idx]
            
            st.metric(
                label="Primary Emotion Detected", 
                value=predicted_class["name"], 
                delta=f"{probabilities[top_idx].item() * 100:.1f}% Confidence"
            )
            
            st.write("---")
            for idx, info in CLASS_MAPPING.items():
                prob_value = probabilities[idx].item()
                st.write(f"{info['name']}")
                st.progress(prob_value)
    
    # --- Occlusion Explainability Feature ---
    st.write("---")
    st.markdown("### 🔍 Explain why the Model made this choice")
    st.markdown("Curious why the model made this prediction? Run an **Occlusion Sensitivity Analysis** to visualize which parts of your dog's face were most critical to the decision.")
    
    if st.button("🔮 Explain Prediction (Generate Heatmap)"):
        with st.spinner("Sliding occlusion windows over features... This takes about 3-5 seconds."):
            
            # Run calculations
            img_resized, heatmap = compute_occlusion_map(model, image, target_class=top_idx)
            
            # Use OpenCV to turn heat matrices into Jet/Thermal color ranges
            heatmap_resized = cv2.resize(heatmap, (img_resized.width, img_resized.height))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Blend original resized PIL image with the colored visual mask
            original_np = np.array(img_resized)
            superimposed_img = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)
            
            # Plot cleanly via Matplotlib
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_resized)
            ax[0].set_title("Original Image (Resized)")
            ax[0].axis("off")
            
            ax[1].imshow(superimposed_img)
            ax[1].set_title(f"X-Ray: Focus on {predicted_class['name']}")
            ax[1].axis("off")
            
            # Render plot into Streamlit container dynamically
            st.pyplot(fig)
            st.success("🔴 **Red/Orange zones** point directly to the pixel areas that heavily drove the chosen emotion choice!")

else:
    st.info("💡 Please upload an image file above to unlock evaluation modes.")