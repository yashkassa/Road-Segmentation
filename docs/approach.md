# Technical Approach — Road Segmentation

## Problem Statement

Road extraction from satellite imagery is a **binary semantic segmentation** task. Each pixel must be classified as either road or non-road. This is harder than it sounds because:

- Roads occupy a very small fraction of the total image area (~5–10%) → severe **class imbalance**
- Roads are **thin and elongated** — standard classification losses blur their edges
- Satellite images have **varying lighting, shadows, occlusions** (trees, buildings)
- Road appearance differs across urban, suburban, and rural regions

---

## Dataset

**DeepGlobe Road Extraction Dataset**  
- High-resolution satellite images (2448 × 2448 px originally)
- Binary masks: white = road, black = background
- Images resized to 512 × 512 for training
- Split: 80% train / 10% validation / 10% test

---

## Architecture Choice: U-Net + ResNet-34

### Why U-Net?
U-Net's encoder-decoder design with **skip connections** is ideal for segmentation tasks where spatial precision matters. The skip connections carry fine-grained spatial information from early encoder layers directly to the decoder — this is critical for detecting thin road structures that would otherwise be lost through repeated downsampling.

### Why ResNet-34 as encoder?
- Pretrained on ImageNet → strong general visual features out of the box
- Relatively lightweight (24M params) — fast to train and fine-tune
- Deep residual connections prevent vanishing gradients during fine-tuning
- Balances accuracy and speed well compared to heavier encoders like ResNet-101 or EfficientNet-B4

---

## Loss Function: Dice + BCE

Standard Binary Cross-Entropy (BCE) alone performs poorly on imbalanced data — the model learns to predict "all background" and still gets low loss. Two losses are combined:

**Dice Loss** directly optimizes the overlap between prediction and ground truth mask, making it robust to class imbalance.

**Soft BCE with Logits** provides stable gradient signal at the pixel level, especially early in training.

```
Total Loss = DiceLoss(pred, target) + SoftBCEWithLogitsLoss(pred, target)
```

This combination consistently outperforms either loss alone for road-like thin-structure segmentation.

---

## Augmentation Strategy

Augmentations prevent overfitting and simulate real-world variation:

| Augmentation | Purpose |
|---|---|
| Horizontal / Vertical Flip | Roads run in all directions |
| Random 90° Rotation | Orientation invariance |
| Shift / Scale / Rotate | Scale and perspective variation |
| Brightness & Contrast | Different lighting / time of day |
| Gaussian Noise | Sensor noise simulation |
| Gaussian Blur | Focus variation |

All augmentations are applied via **Albumentations**, which ensures the mask is transformed identically to the image.

---

## Training Details

- **Optimizer:** AdamW with weight decay (1e-5) to regularize encoder weights
- **LR Schedule:** Cosine Annealing from 1e-4 → 1e-6 over 50 epochs — smooth decay avoids sharp learning rate drops that can destabilize fine-tuned encoders
- **Checkpointing:** Best model saved based on validation IoU
- **Prediction grid:** Saved every 10 epochs for visual inspection of progress

---

## Evaluation Metrics

### IoU (Intersection over Union)
The standard metric for segmentation. Measures how well the predicted mask overlaps with the ground truth.

```
IoU = (Intersection + ε) / (Union + ε)
```

### F1 Score (Dice Coefficient)
Harmonic mean of precision and recall at pixel level. More sensitive to false negatives than IoU, useful for thin structures.

```
F1 = (2 × TP + ε) / (2 × TP + FP + FN + ε)
```

Both metrics are computed per-batch and averaged across the validation set.

---

## Post-Processing

Raw sigmoid outputs are binarized and then cleaned morphologically:

1. **Threshold at 0.5** → binary mask
2. **Morphological Opening** (3×3 kernel) → removes isolated noise pixels (false positives)
3. **Morphological Closing** (3×3 kernel) → fills small gaps within road segments (false negatives)

This step meaningfully improves visual quality of predictions, especially for fragmented road detections.

---

## Potential Improvements

| Idea | Expected Benefit |
|---|---|
| Heavier encoder (EfficientNet-B4, ResNet-50) | Higher accuracy, more parameters |
| Test-Time Augmentation (TTA) | More robust predictions at inference |
| CRF post-processing | Sharper road boundaries |
| Multi-scale inference | Better detection of roads at varying widths |
| Lovász-Softmax loss | Directly optimizes IoU |
