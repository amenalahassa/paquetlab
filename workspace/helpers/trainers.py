import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, valid_loader, scheduler, patience, optimizer, num_epochs = 10):
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = float('inf')
    early_stopping_counter = 0

        # Initialize lists to store metrics
    train_losses, train_accuracies, train_recalls, train_f1_scores = [], [], [], []
    valid_losses, valid_accuracies, valid_recalls, valid_f1_scores = [], [], [], []

    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for data in tqdm(train_loader):
            images = data["image"]
            labels = data["label"]
    
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Append metrics to lists
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)
        
        # Validation
        model.eval()
        valid_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in valid_loader:
                images = data["image"]
                labels = data["label"]
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        valid_loss /= len(valid_loader)
        valid_accuracy = accuracy_score(all_labels, all_preds)
        valid_recall = recall_score(all_labels, all_preds, average='weighted')
        valid_f1 = f1_score(all_labels, all_preds, average='weighted')

        # Append metrics to lists
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_recalls.append(valid_recall)
        valid_f1_scores.append(valid_f1)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, Recall: {valid_recall:.4f}, F1 Score: {valid_f1:.4f}")
        print("-" * 30)
        # Save the model checkpoint if validation loss is improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'models/best_{model.name}_model.pth')
        else:
            early_stopping_counter += 1
                
            
        scheduler.step(valid_loss)

        
        # Early stopping
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    return (train_losses, train_accuracies, train_recalls, train_f1_scores, valid_losses, valid_accuracies, valid_recalls, valid_f1_scores)

def validate_model(model, test_loader, labels_name):
    model.load_state_dict(torch.load(f'models/best_{model.name}_model.pth'))
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            images = data["image"]
            labels = data["label"]
            
            outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_recall_micro = recall_score(all_labels, all_preds, average='micro')
    test_f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}, Micro Recall: {test_recall_micro:.4f}, Micro F1 Score: {test_f1_micro:.4f}")
    
    # Compute metrics per class
    report = classification_report(all_labels, all_preds, target_names=labels_name, output_dict=True)
    
    for class_name, metrics in report.items():
        if class_name not in ('accuracy', 'macro avg', 'weighted avg'):
            print(f"Class: {class_name}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1-score']:.4f}")

def plot_metrics(train_losses, train_accuracies, train_recalls, train_f1_scores, valid_losses, valid_accuracies, valid_recalls, valid_f1_scores):
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, valid_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train')
    plt.plot(epochs, valid_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Plot training and validation recalls
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_recalls, label='Train')
    plt.plot(epochs, valid_recalls, label='Validation')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()
    
    # Plot training and validation F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1_scores, label='Train')
    plt.plot(epochs, valid_f1_scores, label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()