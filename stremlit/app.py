import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
class_names=['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']

class  carClassifierRestNet(nn.Module):
        def __init__(self,dropout_rate=0.6787931076973597):
            super().__init__()
            self.model=models.resnet50(weights='DEFAULT')
            for param in self.model.parameters():
                param.requires_grad=False
            for param in self.model.layer4.parameters():
                param.requires_grad=True
            
            self.model.classifier=nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.model.fc.in_features,6)
            )
        def forward(self,x):
            return self.model(x)
# Placeholder prediction function
def predict_image(img: Image.Image):
    # This is where your model would process the image
    tranforms=transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                                    ])
    # Replace with actual prediction logic
    
    image_tensor=tranforms(img).unsqueeze(0)
    trained_model=carClassifierRestNet()
    trained_model.load_state_dict(torch.load("./model/save_model.pth"))
    trained_model.eval()
    with torch.no_grad():
        output=trained_model(image_tensor)
        _,predicted_class=torch.max(output,1)
    return class_names[predicted_class]
        
        
        

st.title("🐾 Car Prediction App")

# Drag & drop image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Make a prediction
    prediction = predict_image(image)

    st.markdown("### 🔍 Prediction Result")
    st.success(prediction)
