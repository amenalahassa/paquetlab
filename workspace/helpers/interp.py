import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_grad = None
        self.feature_map = None
        self.forward_hook = None
        self.backward_hook = None

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_map = output

        def backward_hook(module, grad_in, grad_out):
            self.feature_grad = grad_out[0]

        self.forward_hook = self.target_layer.register_forward_hook(forward_hook)
        self.backward_hook = self.target_layer.register_backward_hook(backward_hook)

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_cam(self, image_tensor, target_class):
        self.model.zero_grad()
        output = self.model(image_tensor)
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output.to(device))

        weights = torch.mean(self.feature_grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = np.squeeze(cam)
        return cam

# Define a function to generate class activation map
def generate_cam(model, image_tensor, target_class, final_conv_layer):
    model.eval()
    grad_cam = GradCam(model=model, target_layer=final_conv_layer)
    cam = grad_cam.generate_cam(image_tensor, target_class)
    return cam

def display_predicted_cam(model, image, label, transform, id2label, final_conv_layer = None):
    image_tensor = transform(img).to(device).unsqueeze(0)  # Add batch dimension
    
    # Predict the class probabilities
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the predicted class
    predicted_class = torch.argmax(output).item()
    
    # Generate the CAM for the predicted class
    if not final_conv_layer:
        final_conv_layer = model.unet.backbone.layer4
    cam = generate_cam(model, image_tensor, predicted_class, final_conv_layer)

    # Normalize the CAM
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    # Resize the CAM to match the original image size
    image_tsf = image_tensor[0].cpu()
    
    image_np = np.array(image_tsf.permute(1, 2, 0))
    
    # Overlay the CAM on the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(image_np.astype(np.float32), .9, heatmap, 0.5, 0, dtype=cv2.CV_8U)
    
    # Visualize the original image, CAM, and overlay
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(image.permute(1, 2, 0))
    ax[0].set_title(f'Original Image, Original label: {label}')
    ax[0].axis('off')
    # plt.savefig("original.png") 
    ax[1].imshow(image_tsf.permute(1, 2, 0))
    ax[1].set_title('Input Image')
    # ax[1].axis('off')
    # plt.savefig("input.png") 
    cam_im = ax[2].imshow(cam, cmap='jet')
    ax[2].set_title(f'Class Activation Map (CAM), Predicted label: {id2label[predicted_class]}')
    fig.colorbar(cam_im, ax=ax[2], fraction=0.046, pad=0.04)
    # ax[2].axis('off')
    # ax[3].imshow(cam_overlay)
    # ax[3].set_title(f'CAM Overlay, Predicted label: {id2label[predicted_class]}')
    # ax[3].axis('off')
    plt.show()

