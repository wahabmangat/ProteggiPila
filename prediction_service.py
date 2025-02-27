import torch
import torch.nn as nn
from modelClassifier import BoxClassifier
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image


MODEL_PATH_CLASSIFIER="best_loss_modelB.pth"
NUM_CLASSES=2
def predict_classifier(image_path):
    model_classifier = BoxClassifier(num_classes=NUM_CLASSES, pretrained=True, freeze_swin=True)
    model_classifier.load_state_dict(torch.load(MODEL_PATH_CLASSIFIER))  # Load the best model (either accuracy-based or loss-based)
    model_classifier= model_classifier.to("cpu")
    model_classifier.eval()  # Set

    # Define the transformation to match the model's input requirements
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ConvertImageDtype(torch.float),  # Convert to float tensor (0-1 range)

    ])

    # Read and transform the image
    img = read_image(image_path)
    # If the image has 4 channels, remove the 4th channel or convert it to RGB
    if img.shape[0] == 4:
        img = img[:3, :, :]  # Remove the alpha channel (keep only RGB channels)

    image = transform(img).unsqueeze(0).to('cpu')  # Add batch dimension and move to device


    # Make the prediction
    with torch.no_grad():
        output = model_classifier(image)
        output=nn.Softmax()(output)
        _, predicted = torch.max(output, 1)

    return predicted.squeeze().tolist(),output.squeeze().tolist()[-1]

class PredictionService:
  def predict(self, image_path):
      # Returns the following for each frame(image):
      # is_box T/F (don't process further if it isn't box)
      # prediction_confidence_score (prediction_confidence_score in percentage)
      # status (return either 'finished' or 'errored')
      #above_pallet_thresh (T/F)
      pass
