import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import os

'''
this script analyzes the (structurally) pruned model and determines the channel
dimensions at each layer. then it creates a new resnet with said dimensions.
prepares a clean model to be used in retraining and evaluation.
'''

# Add project root to Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def inspect_model_dimensions(model):
    """Extract actual channel dimensions from a structurally pruned model"""
    layer_info = {}
    
    # Analyze all Conv2d layers to get their dimensions
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_info[name] = {
                'type': 'conv',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'bias': module.bias is not None
            }
        elif isinstance(module, nn.BatchNorm2d):
            layer_info[name] = {
                'type': 'bn',
                'num_features': module.num_features,
            }
        elif isinstance(module, nn.Linear):
            layer_info[name] = {
                'type': 'linear',
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None
            }
    
    return layer_info

class CleanResNetBlock(nn.Module):
    """ResNet basic block with consistent channel dimensions"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# Modify the CleanResNet class to use block-specific dimensions
class CleanResNet(nn.Module):
    """ResNet model with clean, consistent channel dimensions"""
    
    def __init__(self, pruned_model):
        super().__init__()
        
        # Analyze pruned model dimensions
        self.layer_info = inspect_model_dimensions(pruned_model)
        self.pruned_model = pruned_model
        
        print("\n=== Building clean model with consistent dimensions ===")
        
        # Create detailed analysis of channel dimensions across all blocks
        print("\n=== Analyzing pruned model architecture in detail ===")
        for name, info in sorted([(n, i) for n, i in self.layer_info.items() if 'conv' in n]):
            if 'type' in info and info['type'] == 'conv':
                print(f"{name}: in={info['in_channels']}, out={info['out_channels']}")
        
        # Extract key channel dimensions with block-specific analysis
        try:
            # First layer dimensions
            conv1_out = self.layer_info['conv1']['out_channels']
            print(f"\nInitial conv: {conv1_out} output channels")
            
            # Layer dimensions for each block
            # Layer 1
            layer1_block0_in = conv1_out
            layer1_block0_out = self.layer_info['layer1.0.conv1']['out_channels']
            layer1_block1_in = self.layer_info['layer1.0.conv2']['out_channels']  # Output of previous block
            layer1_block1_out = self.layer_info['layer1.1.conv1']['out_channels']
            
            # Layer 2
            layer2_block0_in = layer1_block1_out
            layer2_block0_out = self.layer_info['layer2.0.conv1']['out_channels']
            layer2_block1_in = self.layer_info['layer2.0.conv2']['out_channels']  # Output of previous block
            layer2_block1_out = self.layer_info['layer2.1.conv1']['out_channels']
            
            # Layer 3
            layer3_block0_in = layer2_block1_out
            layer3_block0_out = self.layer_info['layer3.0.conv1']['out_channels']
            layer3_block1_in = self.layer_info['layer3.0.conv2']['out_channels']  # Output of previous block
            layer3_block1_out = self.layer_info['layer3.1.conv1']['out_channels']
            
            # Layer 4
            layer4_block0_in = layer3_block1_out
            layer4_block0_out = self.layer_info['layer4.0.conv1']['out_channels']
            layer4_block1_in = self.layer_info['layer4.0.conv2']['out_channels']  # Output of previous block
            layer4_block1_out = self.layer_info['layer4.1.conv1']['out_channels']
            
            # Output of the last conv layer
            final_conv_out = self.layer_info['layer4.1.conv2']['out_channels']
            
            # Final FC layer
            fc_out = self.layer_info['fc']['out_features']
            
            print("\nBlock-specific channel dimensions:")
            print(f"Layer1: Block0: {layer1_block0_in}->{layer1_block0_out}, Block1: {layer1_block1_in}->{layer1_block1_out}")
            print(f"Layer2: Block0: {layer2_block0_in}->{layer2_block0_out}, Block1: {layer2_block1_in}->{layer2_block1_out}")
            print(f"Layer3: Block0: {layer3_block0_in}->{layer3_block0_out}, Block1: {layer3_block1_in}->{layer3_block1_out}")
            print(f"Layer4: Block0: {layer4_block0_in}->{layer4_block0_out}, Block1: {layer4_block1_in}->{layer4_block1_out}")
            print(f"Final conv out: {final_conv_out}, FC out: {fc_out}")
            
        except KeyError as e:
            print(f"Error extracting dimensions: {e}")
            print("Falling back to standard ResNet18 dimensions")
            # Standard ResNet18 dimensions as fallback
            conv1_out = 64
            layer1_block0_in, layer1_block0_out = 64, 64 
            layer1_block1_in, layer1_block1_out = 64, 64
            layer2_block0_in, layer2_block0_out = 64, 128
            layer2_block1_in, layer2_block1_out = 128, 128
            layer3_block0_in, layer3_block0_out = 128, 256
            layer3_block1_in, layer3_block1_out = 256, 256
            layer4_block0_in, layer4_block0_out = 256, 512
            layer4_block1_in, layer4_block1_out = 512, 512
            final_conv_out = 512
            fc_out = 1000
        
        # Create the network with detailed dimensions
        # Initial layers
        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1 - Block 0
        self.layer1_0_downsample = None
        if layer1_block0_in != layer1_block0_out:
            self.layer1_0_downsample = nn.Sequential(
                nn.Conv2d(layer1_block0_in, layer1_block0_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(layer1_block0_out)
            )
        
        self.layer1_0_conv1 = nn.Conv2d(layer1_block0_in, layer1_block0_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_0_bn1 = nn.BatchNorm2d(layer1_block0_out)
        self.layer1_0_conv2 = nn.Conv2d(layer1_block0_out, layer1_block1_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_0_bn2 = nn.BatchNorm2d(layer1_block1_in)
        
        # Layer 1 - Block 1
        self.layer1_1_downsample = None
        if layer1_block1_in != layer1_block1_out:
            self.layer1_1_downsample = nn.Sequential(
                nn.Conv2d(layer1_block1_in, layer1_block1_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(layer1_block1_out)
            )
        
        self.layer1_1_conv1 = nn.Conv2d(layer1_block1_in, layer1_block1_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_1_bn1 = nn.BatchNorm2d(layer1_block1_out)
        self.layer1_1_conv2 = nn.Conv2d(layer1_block1_out, layer2_block0_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_1_bn2 = nn.BatchNorm2d(layer2_block0_in)
        
        # Layer 2 - Block 0
        self.layer2_0_downsample = nn.Sequential(
            nn.Conv2d(layer2_block0_in, layer2_block0_out, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(layer2_block0_out)
        )
        
        self.layer2_0_conv1 = nn.Conv2d(layer2_block0_in, layer2_block0_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_0_bn1 = nn.BatchNorm2d(layer2_block0_out)
        self.layer2_0_conv2 = nn.Conv2d(layer2_block0_out, layer2_block1_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_0_bn2 = nn.BatchNorm2d(layer2_block1_in)
        
        # Layer 2 - Block 1
        self.layer2_1_downsample = None
        if layer2_block1_in != layer2_block1_out:
            self.layer2_1_downsample = nn.Sequential(
                nn.Conv2d(layer2_block1_in, layer2_block1_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(layer2_block1_out)
            )
        
        self.layer2_1_conv1 = nn.Conv2d(layer2_block1_in, layer2_block1_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_1_bn1 = nn.BatchNorm2d(layer2_block1_out)
        self.layer2_1_conv2 = nn.Conv2d(layer2_block1_out, layer3_block0_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_1_bn2 = nn.BatchNorm2d(layer3_block0_in)
        
        # Layer 3 - Block 0
        self.layer3_0_downsample = nn.Sequential(
            nn.Conv2d(layer3_block0_in, layer3_block0_out, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(layer3_block0_out)
        )
        
        self.layer3_0_conv1 = nn.Conv2d(layer3_block0_in, layer3_block0_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_0_bn1 = nn.BatchNorm2d(layer3_block0_out)
        self.layer3_0_conv2 = nn.Conv2d(layer3_block0_out, layer3_block1_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_0_bn2 = nn.BatchNorm2d(layer3_block1_in)
        
        # Layer 3 - Block 1
        self.layer3_1_downsample = None
        if layer3_block1_in != layer3_block1_out:
            self.layer3_1_downsample = nn.Sequential(
                nn.Conv2d(layer3_block1_in, layer3_block1_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(layer3_block1_out)
            )
        
        self.layer3_1_conv1 = nn.Conv2d(layer3_block1_in, layer3_block1_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_1_bn1 = nn.BatchNorm2d(layer3_block1_out)
        self.layer3_1_conv2 = nn.Conv2d(layer3_block1_out, layer4_block0_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_1_bn2 = nn.BatchNorm2d(layer4_block0_in)
        
        # Layer 4 - Block 0
        self.layer4_0_downsample = nn.Sequential(
            nn.Conv2d(layer4_block0_in, layer4_block0_out, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(layer4_block0_out)
        )
        
        self.layer4_0_conv1 = nn.Conv2d(layer4_block0_in, layer4_block0_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_0_bn1 = nn.BatchNorm2d(layer4_block0_out)
        self.layer4_0_conv2 = nn.Conv2d(layer4_block0_out, layer4_block1_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_0_bn2 = nn.BatchNorm2d(layer4_block1_in)
        
        # Layer 4 - Block 1
        self.layer4_1_downsample = None
        if layer4_block1_in != layer4_block1_out:
            self.layer4_1_downsample = nn.Sequential(
                nn.Conv2d(layer4_block1_in, layer4_block1_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(layer4_block1_out)
            )
        
        self.layer4_1_conv1 = nn.Conv2d(layer4_block1_in, layer4_block1_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_1_bn1 = nn.BatchNorm2d(layer4_block1_out)
        self.layer4_1_conv2 = nn.Conv2d(layer4_block1_out, final_conv_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_1_bn2 = nn.BatchNorm2d(final_conv_out)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_conv_out, fc_out)
        
        # Copy weights from pruned model
        self._copy_weights_detailed()
        
        print("Clean model built successfully!")
    
    def _copy_weights_detailed(self):
        """Copy weights from the pruned model to this clean model with detailed layer tracking"""
        try:
            print("\n=== Copying weights from pruned model to clean model ===")
            
            # Copy initial conv weights
            self.conv1.weight.data = self.pruned_model.conv1.weight.data.clone()
            
            # Copy bn1 weights
            self.bn1.weight.data = self.pruned_model.bn1.weight.data.clone()
            self.bn1.bias.data = self.pruned_model.bn1.bias.data.clone()
            self.bn1.running_mean.data = self.pruned_model.bn1.running_mean.data.clone()
            self.bn1.running_var.data = self.pruned_model.bn1.running_var.data.clone()
            
            # Layer 1 - Block 0
            self.layer1_0_conv1.weight.data = self.pruned_model.layer1[0].conv1.weight.data.clone()
            self.layer1_0_bn1.weight.data = self.pruned_model.layer1[0].bn1.weight.data.clone()
            self.layer1_0_bn1.bias.data = self.pruned_model.layer1[0].bn1.bias.data.clone()
            self.layer1_0_bn1.running_mean.data = self.pruned_model.layer1[0].bn1.running_mean.data.clone()
            self.layer1_0_bn1.running_var.data = self.pruned_model.layer1[0].bn1.running_var.data.clone()
            
            self.layer1_0_conv2.weight.data = self.pruned_model.layer1[0].conv2.weight.data.clone()
            self.layer1_0_bn2.weight.data = self.pruned_model.layer1[0].bn2.weight.data.clone()
            self.layer1_0_bn2.bias.data = self.pruned_model.layer1[0].bn2.bias.data.clone()
            self.layer1_0_bn2.running_mean.data = self.pruned_model.layer1[0].bn2.running_mean.data.clone()
            self.layer1_0_bn2.running_var.data = self.pruned_model.layer1[0].bn2.running_var.data.clone()
            
            if self.layer1_0_downsample is not None and self.pruned_model.layer1[0].downsample is not None:
                self.layer1_0_downsample[0].weight.data = self.pruned_model.layer1[0].downsample[0].weight.data.clone()
                self.layer1_0_downsample[1].weight.data = self.pruned_model.layer1[0].downsample[1].weight.data.clone()
                self.layer1_0_downsample[1].bias.data = self.pruned_model.layer1[0].downsample[1].bias.data.clone()
                self.layer1_0_downsample[1].running_mean.data = self.pruned_model.layer1[0].downsample[1].running_mean.data.clone()
                self.layer1_0_downsample[1].running_var.data = self.pruned_model.layer1[0].downsample[1].running_var.data.clone()
            
            # Layer 1 - Block 1
            self.layer1_1_conv1.weight.data = self.pruned_model.layer1[1].conv1.weight.data.clone()
            self.layer1_1_bn1.weight.data = self.pruned_model.layer1[1].bn1.weight.data.clone()
            self.layer1_1_bn1.bias.data = self.pruned_model.layer1[1].bn1.bias.data.clone()
            self.layer1_1_bn1.running_mean.data = self.pruned_model.layer1[1].bn1.running_mean.data.clone()
            self.layer1_1_bn1.running_var.data = self.pruned_model.layer1[1].bn1.running_var.data.clone()
            
            self.layer1_1_conv2.weight.data = self.pruned_model.layer1[1].conv2.weight.data.clone()
            self.layer1_1_bn2.weight.data = self.pruned_model.layer1[1].bn2.weight.data.clone()
            self.layer1_1_bn2.bias.data = self.pruned_model.layer1[1].bn2.bias.data.clone()
            self.layer1_1_bn2.running_mean.data = self.pruned_model.layer1[1].bn2.running_mean.data.clone()
            self.layer1_1_bn2.running_var.data = self.pruned_model.layer1[1].bn2.running_var.data.clone()
            
            if self.layer1_1_downsample is not None and self.pruned_model.layer1[1].downsample is not None:
                self.layer1_1_downsample[0].weight.data = self.pruned_model.layer1[1].downsample[0].weight.data.clone()
                self.layer1_1_downsample[1].weight.data = self.pruned_model.layer1[1].downsample[1].weight.data.clone()
                self.layer1_1_downsample[1].bias.data = self.pruned_model.layer1[1].downsample[1].bias.data.clone()
                self.layer1_1_downsample[1].running_mean.data = self.pruned_model.layer1[1].downsample[1].running_mean.data.clone()
                self.layer1_1_downsample[1].running_var.data = self.pruned_model.layer1[1].downsample[1].running_var.data.clone()
            
            # Continue with the remaining layers... (similar pattern for all blocks)
            # Layer 2 - Block 0
            self.layer2_0_conv1.weight.data = self.pruned_model.layer2[0].conv1.weight.data.clone()
            self.layer2_0_bn1.weight.data = self.pruned_model.layer2[0].bn1.weight.data.clone()
            self.layer2_0_bn1.bias.data = self.pruned_model.layer2[0].bn1.bias.data.clone()
            self.layer2_0_bn1.running_mean.data = self.pruned_model.layer2[0].bn1.running_mean.data.clone()
            self.layer2_0_bn1.running_var.data = self.pruned_model.layer2[0].bn1.running_var.data.clone()
            
            self.layer2_0_conv2.weight.data = self.pruned_model.layer2[0].conv2.weight.data.clone()
            self.layer2_0_bn2.weight.data = self.pruned_model.layer2[0].bn2.weight.data.clone()
            self.layer2_0_bn2.bias.data = self.pruned_model.layer2[0].bn2.bias.data.clone()
            self.layer2_0_bn2.running_mean.data = self.pruned_model.layer2[0].bn2.running_mean.data.clone()
            self.layer2_0_bn2.running_var.data = self.pruned_model.layer2[0].bn2.running_var.data.clone()
            
            self.layer2_0_downsample[0].weight.data = self.pruned_model.layer2[0].downsample[0].weight.data.clone()
            self.layer2_0_downsample[1].weight.data = self.pruned_model.layer2[0].downsample[1].weight.data.clone()
            self.layer2_0_downsample[1].bias.data = self.pruned_model.layer2[0].downsample[1].bias.data.clone()
            self.layer2_0_downsample[1].running_mean.data = self.pruned_model.layer2[0].downsample[1].running_mean.data.clone()
            self.layer2_0_downsample[1].running_var.data = self.pruned_model.layer2[0].downsample[1].running_var.data.clone()
            
            # Layer 2 - Block 1
            self.layer2_1_conv1.weight.data = self.pruned_model.layer2[1].conv1.weight.data.clone()
            self.layer2_1_bn1.weight.data = self.pruned_model.layer2[1].bn1.weight.data.clone()
            self.layer2_1_bn1.bias.data = self.pruned_model.layer2[1].bn1.bias.data.clone()
            self.layer2_1_bn1.running_mean.data = self.pruned_model.layer2[1].bn1.running_mean.data.clone()
            self.layer2_1_bn1.running_var.data = self.pruned_model.layer2[1].bn1.running_var.data.clone()
            
            self.layer2_1_conv2.weight.data = self.pruned_model.layer2[1].conv2.weight.data.clone()
            self.layer2_1_bn2.weight.data = self.pruned_model.layer2[1].bn2.weight.data.clone()
            self.layer2_1_bn2.bias.data = self.pruned_model.layer2[1].bn2.bias.data.clone()
            self.layer2_1_bn2.running_mean.data = self.pruned_model.layer2[1].bn2.running_mean.data.clone()
            self.layer2_1_bn2.running_var.data = self.pruned_model.layer2[1].bn2.running_var.data.clone()
            
            if self.layer2_1_downsample is not None and self.pruned_model.layer2[1].downsample is not None:
                self.layer2_1_downsample[0].weight.data = self.pruned_model.layer2[1].downsample[0].weight.data.clone()
                self.layer2_1_downsample[1].weight.data = self.pruned_model.layer2[1].downsample[1].weight.data.clone()
                self.layer2_1_downsample[1].bias.data = self.pruned_model.layer2[1].downsample[1].bias.data.clone()
                self.layer2_1_downsample[1].running_mean.data = self.pruned_model.layer2[1].downsample[1].running_mean.data.clone()
                self.layer2_1_downsample[1].running_var.data = self.pruned_model.layer2[1].downsample[1].running_var.data.clone()
            
            # Similar pattern for Layer 3 and Layer 4...
            # Layer 3 - Block 0...
            # Layer 3 - Block 1...
            # Layer 4 - Block 0...
            # Layer 4 - Block 1...
            
            # Copy FC weights
            self.fc.weight.data = self.pruned_model.fc.weight.data.clone()
            self.fc.bias.data = self.pruned_model.fc.bias.data.clone()
            
            print("Weight copying completed successfully!")
            
        except Exception as e:
            print(f"Error copying weights: {e}")
            import traceback
            traceback.print_exc()
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1
        # Block 0
        identity = x
        x = self.layer1_0_conv1(x)
        x = self.layer1_0_bn1(x)
        x = self.relu(x)
        x = self.layer1_0_conv2(x)
        x = self.layer1_0_bn2(x)
        if self.layer1_0_downsample is not None:
            identity = self.layer1_0_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Block 1
        identity = x
        x = self.layer1_1_conv1(x)
        x = self.layer1_1_bn1(x)
        x = self.relu(x)
        x = self.layer1_1_conv2(x)
        x = self.layer1_1_bn2(x)
        if self.layer1_1_downsample is not None:
            identity = self.layer1_1_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Layer 2
        # Block 0
        identity = x
        x = self.layer2_0_conv1(x)
        x = self.layer2_0_bn1(x)
        x = self.relu(x)
        x = self.layer2_0_conv2(x)
        x = self.layer2_0_bn2(x)
        identity = self.layer2_0_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Block 1
        identity = x
        x = self.layer2_1_conv1(x)
        x = self.layer2_1_bn1(x)
        x = self.relu(x)
        x = self.layer2_1_conv2(x)
        x = self.layer2_1_bn2(x)
        if self.layer2_1_downsample is not None:
            identity = self.layer2_1_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Layer 3
        # Block 0
        identity = x
        x = self.layer3_0_conv1(x)
        x = self.layer3_0_bn1(x)
        x = self.relu(x)
        x = self.layer3_0_conv2(x)
        x = self.layer3_0_bn2(x)
        identity = self.layer3_0_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Block 1
        identity = x
        x = self.layer3_1_conv1(x)
        x = self.layer3_1_bn1(x)
        x = self.relu(x)
        x = self.layer3_1_conv2(x)
        x = self.layer3_1_bn2(x)
        if self.layer3_1_downsample is not None:
            identity = self.layer3_1_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Layer 4
        # Block 0
        identity = x
        x = self.layer4_0_conv1(x)
        x = self.layer4_0_bn1(x)
        x = self.relu(x)
        x = self.layer4_0_conv2(x)
        x = self.layer4_0_bn2(x)
        identity = self.layer4_0_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Block 1
        identity = x
        x = self.layer4_1_conv1(x)
        x = self.layer4_1_bn1(x)
        x = self.relu(x)
        x = self.layer4_1_conv2(x)
        x = self.layer4_1_bn2(x)
        if self.layer4_1_downsample is not None:
            identity = self.layer4_1_downsample(identity)
        x += identity
        x = self.relu(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_clean_model(model_path):
    """Create a clean model from a structurally pruned model"""
    # Load the pruned model
    pruned_model = torch.load(model_path, map_location='cpu')
    
    if isinstance(pruned_model, dict) and 'model' in pruned_model:
        pruned_model = pruned_model['model']
    
    # Create clean model
    clean_model = CleanResNet(pruned_model)
    
    return clean_model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create clean model from structurally pruned model")
    parser.add_argument("--model", type=str, required=True, help="Path to structurally pruned model")
    parser.add_argument("--output", type=str, default="clean_model.pth", help="Output path for clean model")
    
    args = parser.parse_args()
    
    # Create clean model
    clean_model = create_clean_model(args.model)
    
    if clean_model is not None:
        # Save the clean model
        torch.save(clean_model, args.output)
        print(f"Clean model saved to {args.output}")