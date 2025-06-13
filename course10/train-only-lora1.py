import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    """
    Base neural network - like a standard toolbox
    Simple 3-layer classifier for image classification
    """
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(BaseClassifier, self).__init__()
        
        # Standard layers - your basic tools
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten if needed (for images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass through layers
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        
        x = self.layer3(x)  # No activation - raw logits
        return x

# ============================================================================
# Now let's add LoRA capability
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA adaptation for any Linear layer
    Like adding precision adjustment dials to your existing tools
    """
    def __init__(self, original_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        
        # Keep reference to original layer (frozen)
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get dimensions
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices: like having two adjustment knobs A and B
        # A goes from rank -> out_features (like a volume amplifier)
        # B goes from in_features -> rank (like a frequency selector)
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        
        # Optional dropout for LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Original computation (frozen weights)
        original_output = self.original_layer(x)
        
        # LoRA computation: x -> B -> dropout -> A -> scale
        lora_output = self.dropout(x @ self.lora_B.T)  # Apply B matrix
        lora_output = lora_output @ self.lora_A.T      # Apply A matrix
        lora_output = lora_output * (self.alpha / self.rank)  # Scale
        
        return original_output + lora_output

class ClassifierWithLoRA(BaseClassifier):
    """
    Enhanced classifier with LoRA on specific layers
    Like upgrading specific tools in your toolbox with precision controls
    """
    def __init__(self, input_size=784, hidden_size=512, num_classes=10, 
                 lora_rank=8, lora_alpha=16.0):
        # Initialize the base model first
        super().__init__(input_size, hidden_size, num_classes)
        
        # Now upgrade specific layers with LoRA
        # Layer 1: Add LoRA (like upgrading your main hammer)
        self.layer1 = LoRALayer(
            self.layer1, 
            rank=lora_rank, 
            alpha=lora_alpha,
            dropout=0.1
        )
        
        # Layer 2: Add LoRA (like upgrading your precision screwdriver)
        self.layer2 = LoRALayer(
            self.layer2,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=0.1
        )
        
        # Layer 3: Leave as-is or replace entirely for new task
        # (like keeping your measuring tape the same, or getting a new one)
        
    def freeze_base_parameters(self):
        """Freeze all non-LoRA parameters - lock the original tools"""
        for name, param in self.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = False
                
    def unfreeze_lora_parameters(self):
        """Ensure LoRA parameters are trainable - unlock the adjustment dials"""
        for name, param in self.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
                
    def get_parameter_counts(self):
        """Count parameters like counting tools in your box"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for n, p in self.named_parameters() if 'lora_' in n.lower())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'lora_only': lora_params,
            'efficiency': f"{100 * trainable_params / total_params:.2f}%"
        }

# ============================================================================
# Usage Examples
# ============================================================================

def demonstrate_models():
    """Show the difference between base and LoRA models"""
    
    print("=" * 60)
    print("BASE MODEL (Standard Toolbox)")
    print("=" * 60)
    
    # Create base model
    base_model = BaseClassifier(input_size=784, hidden_size=512, num_classes=10)
    
    # Count parameters
    total_base = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_base:,}")
    
    # Test forward pass
    sample_input = torch.randn(32, 784)  # Batch of 32 flattened 28x28 images
    base_output = base_model(sample_input)
    print(f"Base output shape: {base_output.shape}")
    
    print("\n" + "=" * 60)
    print("LoRA ENHANCED MODEL (Upgraded Toolbox)")
    print("=" * 60)
    
    # Create LoRA model
    lora_model = ClassifierWithLoRA(
        input_size=784, 
        hidden_size=512, 
        num_classes=10,
        lora_rank=8,
        lora_alpha=16.0
    )
    
    # Setup for fine-tuning
    lora_model.freeze_base_parameters()
    lora_model.unfreeze_lora_parameters()
    
    # Get parameter statistics
    stats = lora_model.get_parameter_counts()
    print(f"Total parameters: {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"LoRA parameters only: {stats['lora_only']:,}")
    print(f"Training efficiency: {stats['efficiency']}")
    
    # Test forward pass
    lora_output = lora_model(sample_input)
    print(f"LoRA output shape: {lora_output.shape}")
    
    # Show which parameters are trainable
    print(f"\nTrainable parameters breakdown:")
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name}: {param.shape} ({param.numel():,} params)")

def setup_for_training():
    """Example of setting up LoRA model for training"""
    
    # Create model
    model = ClassifierWithLoRA(lora_rank=4, lora_alpha=8.0)
    
    # Prepare for fine-tuning
    model.freeze_base_parameters()
    model.unfreeze_lora_parameters()
    
    # Create optimizer for only trainable (LoRA) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
    
    return model, optimizer

# Run demonstration
if __name__ == "__main__":
    demonstrate_models()
    
    print(f"\n{'='*60}")
    print("TRAINING SETUP EXAMPLE")
    print(f"{'='*60}")
    
    model, optimizer = setup_for_training()
    print(f"Ready for training with {len(list(model.parameters()))} total params")
    print(f"Optimizer managing {sum(p.numel() for p in optimizer.param_groups[0]['params']):,} trainable params")

