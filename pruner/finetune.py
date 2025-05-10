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
# Default values that can be overridden by command line arguments
default_csv_path = "data/stats/filter_stats.csv"
default_pth_path = "data/models/resnet18.pth"
default_data_dir = "data/10k-imagenet"
default_batch_size = 64
default_num_epochs = 10
default_learning_rate = 1e-3
default_pruning_ratio = 0.1
default_round_to = 8
default_output_dir = "pruned_outputs"

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
                    scores[filter_idx] = float(row["Avg Similarity Score"])

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

def main():
    # --- Parse command line arguments ---
    import argparse
    parser = argparse.ArgumentParser(description="Prune and finetune ResNet model")
    parser.add_argument("--percentage", type=float, default=default_pruning_ratio, help="Percentage of filters to prune (0-1)")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output directory for pruned models")
    parser.add_argument("--csv", type=str, default=default_csv_path, help="Path to filter statistics CSV")
    parser.add_argument("--model", type=str, default=default_pth_path, help="Path to model file")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=default_num_epochs, help="Number of epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=default_learning_rate, help="Learning rate for optimizer")
    parser.add_argument("--round-to", type=int, default=default_round_to, help="Round channels to multiples of this value")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Use the arguments directly instead of global variables
    pruning_ratio = args.percentage
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    round_to = args.round_to
    device = torch.device(args.device)
    # Create output directory with percentage subfolder
    percentage_str = f"{int(pruning_ratio * 100)}"
    output_path = os.path.join(args.output_dir, f"pruned_retrained_{percentage_str}")
    os.makedirs(output_path, exist_ok=True)

    print(f"Pruning {pruning_ratio*100:.1f}% of filters and saving to {output_path}")

    # --- Load pretrained model ---
    model = torchvision.models.resnet18()
    state_dict = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # --- Prune filters ---
    print("Pruning filters...")
    stats_df = load_filter_stats(args.csv)
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

    # Print model infoö before pruning
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
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # For 10k-imagenet, we need to create a train/val split
    # since it only has a single directory
    train_dir = os.path.join(args.data_dir, "imagenet_subtrain")

    # Create a dataset from the train directory
    full_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Use fewer workers to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Fine-tuning ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,              # Start higher to help recovery after pruning
    momentum=0.9,
    weight_decay=5e-4     # Helps generalization
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # --- Save final model ---
    model.zero_grad()  # Clear gradients to reduce file size
    model_save_path = os.path.join(output_path, f"resnet18_pruned_{percentage_str}pct.pth")
    torch.save(model, model_save_path)  # Save the entire model (structure + weights)
    print(f"Model saved to {model_save_path}")

    # --- Plot accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy over Fine-Tuning Epochs ({percentage_str}% Pruning)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, f"accuracy_plot_{percentage_str}pct.png"))

    # Save training history as CSV
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_acc': train_acc_list,
        'val_acc': val_acc_list
    })
    history_df.to_csv(os.path.join(output_path, f"training_history_{percentage_str}pct.csv"), index=False)

    print(f"Training completed for {percentage_str}% pruning. Final validation accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    # This block is required for proper multiprocessing on Windows and macOS
    import multiprocessing
    multiprocessing.freeze_support()
    main()
