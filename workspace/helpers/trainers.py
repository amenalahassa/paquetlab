import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, labels_name, train_loader, valid_loader, scheduler, patience, optimizer, output_dir, num_epochs = 10, average="weighted"):
    best_valid_loss = float('inf')
    early_stopping_counter = 0

        # Initialize lists to store metrics
    train_losses, train_accuracies, train_recalls, train_f1_scores = [], [], [], []
    valid_losses, valid_accuracies, valid_recalls, valid_f1_scores = [], [], [], []
    valid_bal_acc = []
    print(f"Training on {len(train_loader.dataset)} and validating on {len(valid_loader.dataset)} datas")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        all_weights = []

        for data in tqdm(train_loader):
            images = data["image"]
            labels = data["label"]
            weights = data["weight"]
    
            # Forward pass
            outputs = model(images)
            
            if len(labels_name) <= 2:
                criterion = torch.nn.BCEWithLogitsLoss(weight=weights.to(device))
                preds = (torch.sigmoid(outputs) >= 0.5).int().flatten()
                loss = criterion(outputs.flatten(), labels)
                
            else:
                class_weights = train_loader.dataset.get_class_weights()
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
                preds = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, labels.long())
    
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weights.extend(weights.numpy())

        train_loss /= len(train_loader)
        perfo = evaluate_results(labels_name, all_labels, all_preds, average=average, weights=all_weights)

        # Append metrics to lists
        train_losses.append(train_loss)
        train_accuracies.append(perfo["accuracy"])
        train_recalls.append(perfo["recall"])
        train_f1_scores.append(perfo["f1"])
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {perfo['accuracy']:.4f}, Recall: {perfo['recall']:.4f}, F1 Score: {perfo['f1']:.4f}")
        
        valid_metrics = validate_model(model, valid_loader, labels_name, average=average)
        valid_losses.append(valid_metrics["valid_loss"])
        valid_accuracies.append(valid_metrics["accuracy"])
        valid_recalls.append(valid_metrics["recall"])
        valid_f1_scores.append(valid_metrics["f1"])
        valid_bal_acc.append(valid_metrics["bal_acc"])
        
        print("-" * 30)
        
        # Save the model checkpoint if validation loss is improved
        if valid_metrics["valid_loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["valid_loss"]
            torch.save(model.state_dict(), f'{output_dir}/models/best_{model.name}_model.pth')
        else:
            early_stopping_counter += 1
            
        scheduler.step(valid_metrics["valid_loss"])
        
        # Early stopping
        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "train_recalls": train_recalls,
        "train_f1_scores": train_f1_scores,
        
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "valid_recalls": valid_recalls,
        "valid_f1_scores": valid_f1_scores,
        "valid_bal_acc": valid_bal_acc,
    }

    return metrics

def validate_model(model, test_loader, labels_name, average = "micro", plot_confusion = False):
    model.eval()
    all_preds = []
    all_labels = []
    all_weights = []
    valid_loss = 0
    
    print(f"Validating on {len(test_loader.dataset)} datas")
    with torch.no_grad():
        for data in tqdm(test_loader):
            images = data["image"]
            labels = data["label"]
            weights = data["weight"]
            
            outputs = model(images)

            if len(labels_name) <= 2:
                criterion = torch.nn.BCEWithLogitsLoss(weight=weights.to(device))
                preds = (torch.sigmoid(outputs) >= 0.5).int().flatten()
                loss = criterion(outputs.flatten(), labels)
            else:
                class_weights = test_loader.dataset.get_class_weights()
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
                preds = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, labels.long())

            valid_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weights.extend(weights.numpy())

    perfo = evaluate_results(labels_name, all_labels, all_preds, average=average, weights=all_weights)
    print(f"Test Accuracy: {perfo['accuracy']:.4f}, Recall: {perfo['recall']:.4f}, F1 Score: {perfo['f1']:.4f}, Bal Acc Score: {perfo['bal_acc']:.4f}")

    if len(labels_name) > 2:
        for class_name, metrics in perfo["report"].items():
            if class_name not in ('accuracy', 'macro avg', 'weighted avg'):
                print(f"Class: {class_name}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1-score']:.4f}")

    metrics = {
        "accuracy": perfo["accuracy"],
        "recall": perfo["recall"],
        "f1": perfo["f1"],
        "bal_acc": perfo["bal_acc"],
        "valid_loss": valid_loss / len(test_loader)
    }

    if plot_confusion: 
        cm = confusion_matrix(labels, pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
        disp.plot(cmap=plt.cm.Blues)  # You can change the colormap if desired
        plt.title("Confusion Matrix")
        plt.show()
    
    return metrics


def evaluate_results(labels_name, all_labels, all_preds, average="weighted", weights=None):
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_recall_micro = recall_score(all_labels, all_preds, average=average)
    test_f1_micro = f1_score(all_labels, all_preds, average=average)
    bal_acc = balanced_accuracy_score(all_labels, all_preds, sample_weight=weights)
    report = classification_report(all_labels, all_preds, target_names=labels_name, output_dict=True)
    results = {
        "accuracy": test_accuracy,
        "recall": test_recall_micro,
        "f1": test_f1_micro,
        "bal_acc": bal_acc,
        "report": report,
    }
    return results

def plot_metrics(metrics):
    epochs = range(1, len(metrics["train_losses"]) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_losses"], label='Train')
    plt.plot(epochs, metrics["valid_losses"], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_accuracies"], label='Train')
    plt.plot(epochs, metrics["valid_accuracies"], label='Validation')
    plt.plot(epochs, metrics["valid_bal_acc"], label='Validation - Balanced Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    # Plot training and validation recalls
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_recalls"], label='Train')
    plt.plot(epochs, metrics["valid_recalls"], label='Validation')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()
    
    # Plot training and validation F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics["train_f1_scores"], label='Train')
    plt.plot(epochs, metrics["valid_f1_scores"], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()