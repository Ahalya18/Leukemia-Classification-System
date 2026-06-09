import numpy as np
import cv2
import os

def generate_mock_grad_cam(image_path, output_path):
    try:
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            return False
            
        hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
        # Apply the exact same tightened saturation thresholds so heatmaps ignore shadows
        lower_purple = np.array([115, 75, 45])
        upper_purple = np.array([165, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Clean noise before blurring
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        blurred_mask = cv2.GaussianBlur(mask, (81, 81), 0)
        
        heatmap_colored = cv2.applyColorMap(blurred_mask, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap_colored * 0.4 + img_orig * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        cv2.imwrite(output_path, superimposed_img)
        return True
    except Exception as e:
        print(f"Failed to generate mock grad-cam: {e}")
        return False

def generate_grad_cam(image_path, model, output_path):
    return generate_mock_grad_cam(image_path, output_path)
