# medichain-fl/backend/model.py

import torch
import torch.nn as nn
# IMPORTANT CHANGE: Reverted to AutoModelForImageClassification as dima806 model is PyTorch native
from transformers import AutoModelForImageClassification, AutoImageProcessor
# Removed hf_hub_download as AutoModel handles it for standard PyTorch models

def load_model(model_name: str = "dima806/chest_xray_pneumonia_detection", freeze_encoder: bool = True) -> nn.Module:
    """
    Loads a pre-trained Hugging Face PyTorch model for image classification.
    The model_name is set to "dima806/chest_xray_pneumonia_detection".
    Optionally freezes all layers except the classification head for transfer learning.
    """
    print(f"Loading Hugging Face model: {model_name}")
    
    # IMPORTANT CHANGE: Use AutoModelForImageClassification.from_pretrained for PyTorch model
    # This automatically loads the correct model architecture (e.g., a custom CNN)
    # and its weights (from model.safetensors or pytorch_model.bin)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Freeze encoder layers (all layers except the classification head)
    if freeze_encoder:
        print("Freezing all model parameters except the classification head.")
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classification head.
        # For this model, the final layer is usually 'classifier'.
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            for param in model.classifier.parameters():
                param.requires_grad = True
            print("Unfrozen 'classifier' head (final classification layer).")
        # Added a check for 'fc' which is common in some ResNet-like custom heads
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
             for param in model.fc.parameters():
                param.requires_grad = True
             print("Unfrozen 'fc' layer.")
        else:
            print("Warning: Could not automatically detect and unfreeze classification head. "
                  "Manual inspection of model architecture might be needed.")

    # Adapt the classification head if the number of classes differs.
    # The 'dima806/chest_xray_pneumonia_detection' is likely for 2 classes.
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        if model.classifier.out_features != 2: # Assuming 2 classes: NORMAL, PNEUMONIA
            original_in_features = model.classifier.in_features
            model.classifier = nn.Linear(original_in_features, 2)
            print(f"Adapted classifier head to 2 output features (from {model.classifier.out_features}).")
            # Ensure the newly created classifier is trainable
            for param in model.classifier.parameters():
                param.requires_grad = True
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Module): # Check 'fc' if 'classifier' not found
         if model.fc.out_features != 2:
            original_in_features = model.fc.in_features
            model.fc = nn.Linear(original_in_features, 2)
            print(f"Adapted 'fc' head to 2 output features (from {model.fc.out_features}).")
            for param in model.fc.parameters():
                param.requires_grad = True


    # Ensure the model is set to training mode by default for initial setup
    model.train()
    
    return model

def load_image_processor(model_name: str = "dima806/chest_xray_pneumonia_detection"):
    """
    Loads the corresponding image processor for a Hugging Face model.
    """
    print(f"Loading Hugging Face image processor for: {model_name}")
    return AutoImageProcessor.from_pretrained(model_name)

if __name__ == '__main__':
    # Simple test to check model loading and output shape
    # IMPORTANT CHANGE: Default model_name is now dima806
    model = load_model()
    processor = load_image_processor()

    # Create a dummy grayscale image (PIL Image)
    from PIL import Image
    dummy_image = Image.new('L', (224, 224)) # 'L' for grayscale
    
    # Process the image to get model inputs (Tensor)
    # The processor will handle channel replication if the model expects RGB
    inputs = processor(images=dummy_image, return_tensors="pt")
    
    print(f"Processed input pixel values shape: {inputs['pixel_values'].shape}")

    # Perform a forward pass
    model.eval() # Set to eval mode for inference test
    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        # dima806 model outputs directly return logits in the first position if it's not a standard transformers output object
        # If it returns a standard output object, outputs.logits is correct.
        # Let's assume it returns outputs.logits for compatibility with AutoModelForImageClassification
        logits = outputs.logits
        print(f"Model output logits shape: {logits.shape}") # Should be torch.Size([1, 2])
    print(model)