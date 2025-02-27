import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LocationDataset  # Ensure LocationDataset is correctly implemented
from model import LocationClassifier  # Ensure LocationClassifier is correctly implemented

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define dataset paths
DATA_PATH = "/home/joaquinecc/ailivesim/datasets/small_location_datasett"
BATCH_SIZE = 4
NUM_CLASSES = 5  # Modify according to your dataset
EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming the rest of your code (Dataset, Model, etc.) is correct

# Initialize the model, criterion, optimizer, and dataloaders as in your existing code

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        val_acc = evaluate_model(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

# Evaluation function including classification report
def evaluate_model_test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    
    # Print classification report (precision, recall, F1-score)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes=[str(i) for i in range(NUM_CLASSES)], normalize=False)
    
    return accuracy, avg_loss

# Evaluation function
def evaluate_model(model, loader,criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return accuracy, avg_loss


def test_best_model(model, test_loader, model_path,criterion):
    model.load_state_dict(torch.load(model_path))  # Load the best model (either accuracy-based or loss-based)
    test_acc = evaluate_model(model, test_loader,criterion)[0]
    print(f"Test Accuracy with {model_path}: {test_acc:.4f}")
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_loss_model_path = "best_loss_modelB.pth"
    best_acc_model_path = "best_modelB.pth"

    best_val_loss = float('inf')  # Initialize with a very high loss value
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        val_acc ,val_loss= evaluate_model(model, val_loader,criterion)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_acc_model_path)
            print("Best model saved based on accuracy.")

        # Save the best model based on lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), best_loss_model_path)
            print("Best model saved based on lowest loss.")

