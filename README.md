# CIFAR-10 Image Classification with ResNet50V2

A deep learning project exploring transfer learning, fine-tuning, and architectural optimization for image classification on the CIFAR-10 dataset. The project follows an iterative research cycle — from a frozen-weight baseline to a fully fine-tuned model — achieving a final test accuracy of **93.6%**.

---

## Project Overview

CIFAR-10 is a benchmark dataset of 60,000 low-resolution (32×32) images across 10 classes. The core challenge is that state-of-the-art CNN architectures like ResNet are designed for much larger inputs (224×224), making naive application ineffective.

This project addresses that challenge through systematic architectural modifications and regularization techniques, documenting the reasoning and impact of each change.

---

## Results Summary

| Model | Test Accuracy | Key Issue |
|---|---|---|
| Baseline (frozen ResNet50V2) | 66.0% | Severe overfitting |
| Enhanced (fine-tuned + regularized) | **93.6%** | Resolved |

---

## Methodology

### Baseline Model
- Loaded ResNet50V2 with pretrained ImageNet weights (`weights='imagenet'`)
- Froze all base model layers for pure feature extraction
- Added a Global Average Pooling layer + single Dense output head
- Applied 2x upsampling to handle the 32×32 → ResNet resolution mismatch

**Diagnosis:** Training curves showed the model began overfitting after just 3 epochs. Classification report revealed the worst performance on visually similar classes (Cat: 51% precision, Bird: 53% recall), indicating the frozen ImageNet features were insufficient for CIFAR-10's low-resolution textures.

### Enhanced Model
Based on the baseline diagnosis, the following modifications were made:

**1. Transfer Learning → Fine-Tuning**  
Unfroze all ResNet50V2 layers and applied a low learning rate (`1e-4` with `ReduceLROnPlateau`) to surgically adapt pretrained ImageNet weights to CIFAR-10 textures without destroying learned representations.

**2. Resolution Fix via 4x Upsampling**  
Increased input resolution from 32×32 to 128×128 using `UpSampling2D(4,4)`, preventing feature maps from collapsing to 1×1 too early in the ResNet blocks.

**3. Intensive Data Augmentation**  
Added a preprocessing pipeline with `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, and `RandomContrast` to artificially expand dataset variety and force the model to learn invariant features rather than memorizing pixel positions.

**4. Regularized Classification Head**  
Replaced the single Dense layer with a deeper head:
- Dense(512) → BatchNormalization → Dropout(0.4)
- Dense(256) → BatchNormalization → Dropout(0.2)
- Dense(10, softmax)

**5. Label Smoothing**  
Used `CategoricalCrossentropy(label_smoothing=0.1)` to prevent overconfident predictions and improve generalization.

---

## Key Findings

- The most impactful single change was **unfreezing the base model for fine-tuning** — frozen ImageNet weights alone could not capture CIFAR-10's low-resolution texture patterns
- **Data augmentation** was essential for a dataset this small; without it, ResNet50V2's capacity causes near-immediate memorization
- **Global Average Pooling** outperformed Flatten by reducing parameters and improving spatial invariance
- Hardest classes remained **Cat vs. Dog** due to visual similarity at low resolution, though both improved significantly (Cat F1: 51% → 87%, Dog F1: improved to 89%)

---

## Tech Stack

- Python, TensorFlow/Keras
- ResNet50V2 (ImageNet pretrained)
- Scikit-learn (evaluation metrics)
- Google Colab (GPU training)
- Google Drive (model checkpointing)

---

## File Structure
├── assignment2_cnn.ipynb     # Full notebook with code and analysis
├── README.md
└── models/                   # Saved .keras model checkpoints
├── resnet50_baseline.keras
└── resnet50_enhanced.keras

---

## Lessons Learned

This project reinforced that deep learning development is inherently iterative. The final 93.6% accuracy was not a first attempt — it was the result of using evaluation metrics as a diagnostic tool, identifying specific failure points (overfitting, resolution mismatch, insufficient regularization), and redesigning the architecture to address each one systematically.
