import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
)
from tqdm import tqdm

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion,
                       num_epochs=10, device='cuda', is_binary=True):
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_f1': [],
        'val_f1': [],
        'train_auc': [],
        'val_auc': [],
        'train_cm': [],
        'val_cm': []
    }

    best_val_f1 = -float('inf')
    best_model = None
    best_epoch = -1

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []

        for inputs_title, inputs_content, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs_title, inputs_content, targets = inputs_title.to(device), inputs_content.to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs_title, inputs_content)
            # print("OUTPUTS:", outputs)

            if is_binary:
                if outputs.shape != targets.shape:
                    if outputs.dim() > targets.dim():
                        targets = targets.view(-1, 1)
                    else:
                        outputs = outputs.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if is_binary:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_targets.extend(targets.detach().cpu().numpy())

        all_train_preds = np.array(all_train_preds).flatten()
        all_train_targets = np.array(all_train_targets).flatten()
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_train_targets, all_train_preds, average='binary' if is_binary else 'weighted', zero_division=0
        )
        train_cm = confusion_matrix(all_train_targets, all_train_preds)

        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for inputs_title, inputs_content, targets in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs_title, inputs_content, targets = inputs_title.to(device), inputs_content.to(device), targets.float().to(device)
                # inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs_title, inputs_content)

                if is_binary:
                    if outputs.shape != targets.shape:
                        if outputs.dim() > targets.dim():
                            targets = targets.view(-1, 1)
                        else:
                            outputs = outputs.view(-1)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

                if is_binary:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.detach().cpu().numpy())
                all_val_targets.extend(targets.detach().cpu().numpy())

        all_val_preds = np.array(all_val_preds).flatten()
        all_val_targets = np.array(all_val_targets).flatten()
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_val_targets, all_val_preds, average='binary' if is_binary else 'weighted', zero_division=0
        )
        val_cm = confusion_matrix(all_val_targets, all_val_preds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_cm'].append(train_cm)
        history['val_cm'].append(val_cm)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}')
        print(f'  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            best_epoch = epoch
            print(f'  New best model saved! Validation F1: {val_f1:.4f}')

    model.load_state_dict(best_model)
    print(f'Best model from epoch {best_epoch+1} with Validation F1: {best_val_f1:.4f}')
    return history, model

def plot_metrics(history, figsize=(20, 15), class_names=None):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=figsize)

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_precision'], 'b-', label='Training Precision')
    plt.plot(epochs, history['val_precision'], 'r-', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_recall'], 'b-', label='Training Recall')
    plt.plot(epochs, history['val_recall'], 'r-', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['train_f1'], 'b-', label='Training F1')
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('metrics_plot_llmic.png')
    

    if 'train_cm' in history and 'val_cm' in history:
        best_epoch = np.argmax(history['val_f1'])
        best_val_f1 = history['val_f1'][best_epoch]
        print(f"CF for the best model (Epoch {best_epoch+1}, Validation F1: {best_val_f1:.4f})")

        if class_names is None:
            n_classes = history['val_cm'][best_epoch].shape[0]
            class_names = [str(i) for i in range(n_classes)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=history['train_cm'][best_epoch],
            display_labels=class_names
        ).plot(ax=ax1, cmap='Blues', values_format='d')
        ax1.set_title(f'Training Confusion Matrix (Epoch {best_epoch+1})')

        ConfusionMatrixDisplay(
            confusion_matrix=history['val_cm'][best_epoch],
            display_labels=class_names
        ).plot(ax=ax2, cmap='Blues', values_format='d')
        ax2.set_title(f'Validation Confusion Matrix (Epoch {best_epoch+1})')

        plt.tight_layout()
        plt.savefig('confusion_matrix_plot_llmic.png')

def plot_confusion_matrix_evolution(history, figsize=(15, 10), class_names=None, epochs_to_plot=None):
    if 'val_cm' not in history:
        print("No confusion matrices available in history")
        return

    if epochs_to_plot is None:
        total_epochs = len(history['val_cm'])
        if total_epochs <= 3:
            epochs_to_plot = list(range(total_epochs))
        else:
            epochs_to_plot = [0, total_epochs // 2, total_epochs - 1]

    epochs_to_plot = [e for e in epochs_to_plot if e < len(history['val_cm'])]

    if class_names is None:
        n_classes = history['val_cm'][0].shape[0]
        class_names = [str(i) for i in range(n_classes)]

    n_epochs = len(epochs_to_plot)
    fig, axes = plt.subplots(2, n_epochs, figsize=figsize)
    for i, epoch_idx in enumerate(epochs_to_plot):
        epoch_num = epoch_idx + 1
        if n_epochs == 1:
            ax_train = axes[0]
            ax_val = axes[1]
        else:
            ax_train = axes[0, i]
            ax_val = axes[1, i]

        ConfusionMatrixDisplay(
            confusion_matrix=history['train_cm'][epoch_idx],
            display_labels=class_names
        ).plot(ax=ax_train, cmap='Blues', values_format='d')
        ax_train.set_title(f'Training CM - Epoch {epoch_num}')

        ConfusionMatrixDisplay(
            confusion_matrix=history['val_cm'][epoch_idx],
            display_labels=class_names
        ).plot(ax=ax_val, cmap='Blues', values_format='d')
        ax_val.set_title(f'Validation CM - Epoch {epoch_num}')

    plt.tight_layout()
    plt.savefig('confusion_matrix_evolution_plot_llmic.png')


def evaluate_on_threshold(probs, targets, threshold):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    return acc, prec, rec, f1

def get_test_probs_targets(model, test_loader, device='cuda'):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs_title, inputs_content, targets in tqdm(test_loader, desc='Testing'):
            # Ensure 2D shape even for batch_size=1
            if inputs_title.dim() == 1:
                inputs_title = inputs_title.unsqueeze(0)
            if inputs_content.dim() == 1:
                inputs_content = inputs_content.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            inputs_title = inputs_title.to(device)
            inputs_content = inputs_content.to(device)
            targets = targets.float().to(device)

            outputs = model(inputs_title, inputs_content)
            probs = torch.sigmoid(outputs).view(-1)
            targets = targets.view(-1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_targets)
