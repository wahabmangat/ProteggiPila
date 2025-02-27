import torch
import torch.nn as nn
from model.modelClassifier import BoxClassifier
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image



from ultralytics import YOLO
import os

MODEL_PATH_CLASSIFIER = "models/classifier_model.pth"
NUM_CLASSES = 2
YOLO_MODEL = "models/yolov8n_cardboard_50_epochs_trained.pt"


class PredictionService:
    def __init__(self):
        # Load the trained model with error handling
        try:
            self.trained_model = YOLO(YOLO_MODEL)
        except Exception as e:
            self._handle_error(f"Error loading YOLO model: {str(e)}")

        try:
            self.model_classifier = BoxClassifier(num_classes=NUM_CLASSES, pretrained=True, freeze_swin=True)
            self.model_classifier.load_state_dict(torch.load(MODEL_PATH_CLASSIFIER))  # Load the best model
            self.model_classifier = self.model_classifier.to("cpu")
            self.model_classifier.eval()  # Set the model to evaluation mode
        except Exception as e:
            self._handle_error(f"Error loading classifier model: {str(e)}")

        # Define the transformation to match the model's input requirements
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ConvertImageDtype(torch.float),  # Convert to float tensor (0-1 range)
        ])

    def _classifie_prediction(self, image_path):
        try:
            # Read and transform the image
            img = read_image(image_path)
            # If the image has 4 channels, remove the 4th channel or convert it to RGB
            if img.shape[0] == 4:
                img = img[:3, :, :]  # Remove the alpha channel (keep only RGB channels)

            image = self.transform(img).unsqueeze(0).to('cpu')  # Add batch dimension and move to device

            # Make the prediction
            with torch.no_grad():
                output = self.model_classifier(image)
                output = nn.Softmax()(output)
                _, predicted = torch.max(output, 1)

            return predicted.squeeze().tolist(), output.squeeze().tolist()[-1]
        except Exception as e:
            self._handle_error(f"Error during classification: {str(e)}")

    def _object_predict(self, image_path):
        try:
            # Perform inference on the image
            results = self.trained_model(image_path)
            
            # Initialize default values
            is_box = False
            
            # Check if any boxes are detected
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    is_box = True
                    break
            
            return is_box
        except Exception as e:
            self._handle_error(f"Error during object detection: {str(e)}")

    def _handle_error(self, error_message):
        # Handle errors by printing the error message and setting response status to 'errored'
        print(f"Error: {error_message}")
        raise Exception(error_message)

    def predict(self, image_path):
        # Returns the following for each frame(image):
        # is_box T/F (don't process further if it isn't a box)
        # prediction_confidence_score (prediction_confidence_score in percentage)
        # status (return either 'finished' or 'errored')
        # above_pallet_thresh (T/F)
        response = {
            "is_box": False,
            "prediction_confidence_score": None,
            "status": None,
            "above_pallet_thresh": None,
        }

        try:
            is_box = self._object_predict(image_path)
            response["is_box"] = is_box
            
            if is_box:
                above_pallet_thresh, prediction_confidence_score = self._classifie_prediction(image_path)
                response["above_pallet_thresh"] = True if above_pallet_thresh else False
                response["prediction_confidence_score"] = prediction_confidence_score

            response["status"] = "finished"
        except Exception as e:
            response["status"] = "errored"
            response["error_message"] = str(e)

        return response


MODEL_PATH_CLASSIFIER="checkpoints/classifier_model.pth"
NUM_CLASSES=2
YOLO_MODEL="checkpoints/yolov8n_cardboard_50_epochs_trained.pt"


class PredictionService:
    def __init__(self):
      # Load the trained model
      self.trained_model = YOLO(YOLO_MODEL)

      self.model_classifier = BoxClassifier(num_classes=NUM_CLASSES, pretrained=True, freeze_swin=True)
      self.model_classifier.load_state_dict(torch.load(MODEL_PATH_CLASSIFIER))  # Load the best model (either accuracy-based or loss-based)
      self.model_classifier= self.model_classifier.to("cpu")
      self.model_classifier.eval()  # Set

      # Define the transformation to match the model's input requirements
      self.transform = transforms.Compose([
          transforms.Resize((224, 224)),  # Resize to 224x224
          transforms.ConvertImageDtype(torch.float),  # Convert to float tensor (0-1 range)

      ])

    def _classifie_prediction(self,image_path):
      # Read and transform the image
      img = read_image(image_path)
      # If the image has 4 channels, remove the 4th channel or convert it to RGB
      if img.shape[0] == 4:
          img = img[:3, :, :]  # Remove the alpha channel (keep only RGB channels)

      image = self.transform(img).unsqueeze(0).to('cpu')  # Add batch dimension and move to device


      # Make the prediction
      with torch.no_grad():
          output = self.model_classifier(image)
          output=nn.Softmax()(output)
          _, predicted = torch.max(output, 1)

      return predicted.squeeze().tolist(),output.squeeze().tolist()[-1]

    def _object_predict(self, image_path):
        # Perform inference on the image
        results = self.trained_model(image_path)
        
        # Initialize default values
        is_box = False
        
        # Check if any boxes are detected
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                is_box = True
                break
        
        return is_box
    
    def predict(self, image_path):
      # Returns the following for each frame(image):
      # is_box T/F (don't process further if it isn't box)
      # prediction_confidence_score (prediction_confidence_score in percentage)
      # status (return either 'finished' or 'errored')
      #above_pallet_thresh (T/F)
      response={
          "is_box":False,
          "prediction_confidence_score":None,
          "status":None,
          "above_pallet_thresh":None

      }
      is_box=self._object_predict(image_path)
      response["is_box"]=is_box
      if is_box:
        above_pallet_thresh,prediction_confidence_score=self._classifie_prediction(image_path)
        response["above_pallet_thresh"]=True if above_pallet_thresh else False
        response["prediction_confidence_score"]=prediction_confidence_score

      
      response["status"]="finished"
      return response
