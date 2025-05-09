import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch_pruning as tp
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- CONFIG ---
csv_path = "data/filter_stats (2).csv"
pth_path = "data/models/resnet18.pth"
data_dir = "data/tiny-imagenet-200"
batch_size = 64
num_epochs = 10
learning_rate = 1e-3
pruning_ratio = 0.3  # Percentage of filters to prune
round_to = 8  # Round channels to multiples of 8 for better hardware acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load CSV ---
def load_filter_stats(stats_file):
    print(f"Loading filter statistics from {stats_file}")
    try:
        stats_df = pd.read_csv(stats_file)
        print(f"Loaded statistics for {len(stats_df)} filters")
        return stats_df
    except Exception as e:
        print(f"Error loading filter statistics: {e}")
        return None

def create_importance_scores_from_csv(stats_df, model):
    """Create a dictionary mapping layers to their filter importance scores"""
    required_columns = ["Layer", "Filter", "Avg Similarity Score"]
    alt_columns = {"Layer": "layer_id", "Filter": "filter_id", "Avg Similarity Score": "avg_similarity_score"}

    # Rename columns if needed
    for req_col, alt_col in alt_columns.items():
        if req_col not in stats_df.columns and alt_col in stats_df.columns:
            stats_df = stats_df.rename(columns={alt_col: req_col})

    if any(col not in stats_df.columns for col in required_columns):
        print("Missing required columns after rename.")
        return {}

    # Get all Conv2d layers
    conv_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]

    # Create importance scores dictionary
    importance_dict = {}

    for layer_idx, layer_df in stats_df.groupby("Layer"):
        if layer_idx < len(conv_layers):
            _, module = conv_layers[layer_idx]
            # Create tensor of importance scores for this layer
            scores = torch.ones(module.out_channels)

            # Fill in the scores from the dataframe
            for _, row in layer_df.iterrows():
                filter_idx = int(row["Filter"])
                if filter_idx < module.out_channels:
                    # Lower score means higher importance (we'll use 1 - similarity)
                    scores[filter_idx] = 1.0 - float(row["Avg Similarity Score"])

            importance_dict[module] = scores

    return importance_dict

# --- Custom importance criterion based on CSV data ---
class CSVImportance(tp.importance.Importance):
    def __init__(self, importance_dict):
        super().__init__()
        self.importance_dict = importance_dict

    def __call__(self, group):
        # Get the root module from the group
        root_module = group[0][0].target.module

        if root_module in self.importance_dict:
            # Get the importance scores for this module
            scores = self.importance_dict[root_module]
            # Return the scores for the channels in this group
            return scores
        else:
            # If no scores available, return ones (neutral importance)
            return torch.ones(root_module.out_channels)

def main():
    # --- Load pretrained model ---
    model = torchvision.models.resnet18()
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # --- Prune filters ---
    print("Pruning filters...")
    stats_df = load_filter_stats(csv_path)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # Create importance scores from CSV
    importance_dict = create_importance_scores_from_csv(stats_df, model)
    importance_criterion = CSVImportance(importance_dict)

    # Identify layers to ignore (e.g., final classifier)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)  # Don't prune the final classifier

    # Initialize pruner with the model and importance criterion
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance_criterion,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
    )

    # Print model info before pruning
    base_stats = tp.utils.count_ops_and_params(model, example_inputs)
    base_macs, base_nparams = base_stats[0], base_stats[1]
    print(f"Before pruning: {base_macs/1e9:.2f} GMACs, {base_nparams/1e6:.2f}M parameters")

    # Perform pruning
    pruner.step()

    # Print model info after pruning
    stats = tp.utils.count_ops_and_params(model, example_inputs)
    macs, nparams = stats[0], stats[1]
    print(f"After pruning: {macs/1e9:.2f} GMACs, {nparams/1e6:.2f}M parameters")
    print(f"Reduction: {(1 - macs/base_macs)*100:.2f}% MACs, {(1 - nparams/base_nparams)*100:.2f}% parameters")

    # --- Dataloaders (upsample to 224x224) ---
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    # Use fewer workers to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Fine-tuning ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_acc_list, val_acc_list = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # --- Save final model ---
    model.zero_grad()  # Clear gradients to reduce file size
    torch.save(model, "resnet18_pruned_finetuned.pth")  # Save the entire model (structure + weights)
    print("Model saved to resnet18_pruned_finetuned.pth")

    # --- Plot accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Fine-Tuning Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    # This block is required for proper multiprocessing on Windows and macOS
    import multiprocessing
    multiprocessing.freeze_support()
    main()
