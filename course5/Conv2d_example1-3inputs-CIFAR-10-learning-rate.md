Great observation! The learning rate is reduced by half at epoch 5 (halfway through the 10 epochs) due to this line in the code:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```

This is a **learning rate scheduling strategy** - let me explain why this is beneficial:

## **Why Reduce Learning Rate During Training?**

**The "Coarse-to-Fine" Learning Analogy:**

Think of learning rate like adjusting the size of steps when you're trying to find the bottom of a valley:

1. **Early Training (High LR)**: Take big steps to quickly get into the general area of the minimum
2. **Later Training (Lower LR)**: Take smaller, more careful steps to precisely find the exact minimum

## **The Technical Reasoning:**

**Early Training (Epochs 1-5, LR = 0.001):**
- **Large gradients**: Network is far from optimal, needs big updates
- **Fast convergence**: Get to the general vicinity of good weights quickly
- **Coarse adjustments**: Don't need precision yet

**Later Training (Epochs 6-10, LR = 0.0005):**
- **Smaller gradients**: Network is closer to optimal, needs fine-tuning
- **Stability**: Prevents overshooting the minimum
- **Fine adjustments**: Polish the weights for better accuracy

## **What Happens Without LR Scheduling:**

```python
# Bad: Fixed learning rate
# Early: Too slow to converge (wastes time)
# Later: Too aggressive, keeps bouncing around minimum (never settles)

# Good: Scheduled learning rate  
# Early: Fast convergence to good region
# Later: Fine-tuning for optimal performance
```

## **Visual Analogy:**

```
Loss Landscape (like a valley):

Fixed High LR:     🏃‍♂️💨 → 🏃‍♂️💨 → 🏃‍♂️💨 (keeps running past the target)
Fixed Low LR:      🚶‍♂️    → 🚶‍♂️    → 🚶‍♂️    (too slow to get there)
Scheduled LR:      🏃‍♂️💨 → 🏃‍♂️💨 → 🚶‍♂️    → 🚶‍♂️  (fast start, careful finish)
```

## **Common LR Scheduling Strategies:**

1. **StepLR** (what we used): Reduce by factor every N epochs
2. **CosineAnnealingLR**: Gradually decrease following cosine curve
3. **ReduceLROnPlateau**: Reduce when validation loss stops improving
4. **ExponentialLR**: Exponential decay

## **The CIFAR-10 Specific Reason:**

CIFAR-10 is more complex than MNIST:
- **More classes** (10 vs 10, but more visually similar)
- **Color information** to learn
- **Complex textures** and shapes
- **Needs more careful optimization**

## **You Could Experiment With:**

```python
# Option 1: No scheduling (see the difference!)
# scheduler = None  # Comment out scheduler.step()

# Option 2: More aggressive reduction
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

# Option 3: Multiple reductions
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7], gamma=0.5)

# Option 4: Smooth reduction
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

## **The Result:**

You'll typically see:
- **Epochs 1-5**: Loss drops quickly (big steps)
- **Epoch 5**: Learning rate cuts in half
- **Epochs 6-10**: Loss drops more slowly but steadily (fine-tuning)
- **Final result**: Better convergence than fixed LR

This is a **standard best practice** in deep learning - almost all state-of-the-art models use some form of learning rate scheduling. It's like shifting gears in a car: start in high gear for speed, then shift to lower gear for precision!

Try running the code with and without the scheduler to see the difference in training curves!

