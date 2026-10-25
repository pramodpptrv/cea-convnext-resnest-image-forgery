# Image Forgery Classification Using Compression Error Analysis and ConvNeXt–ResNeSt


---

## Highlights

- **CEA input:** JPEG recompression at quality 90, absolute difference, contrast scaling to 0–255.
- **Backbone and refinement:** ConvNeXt-Tiny followed by a single ResNeSt split-attention block (K=1, radix=2).
- **Two-stage training:** 25 epochs with a frozen backbone, then 100 epochs of full fine-tuning at a 10x lower learning rate.
- **Leakage-aware evaluation:** a source- and duplicate-aware split of CASIA v2.0 (SHA-256 + source IDs + perceptual hashing).

---

## Pipeline

```
Image -> CEA (JPEG Q=90, |I - J90(I)|, scale to 0-255) -> resize 160x160x3
      -> ConvNeXt-Tiny (ImageNet weights) -> 5x5x768 feature map
      -> ResNeSt block (1x1 768->1536, 3x3 grouped conv, radix-2 split attention, 1x1 ->768, residual)
      -> GAP -> Dropout(0.5) -> Dense(1) -> sigmoid -> authentic / tampered (threshold 0.5)
```

---

## Datasets

All datasets are public benchmarks. The copies used here come from Kaggle mirrors. The Kaggle pages list the license as "Unknown", so please respect the terms of the original releases.

| Dataset | Role | Images used | Authentic | Tampered | Notes |
|---|---|---|---|---|---|
| CASIA v2.0 | Training / validation / test | 2,329 (subset) | 1,171 | 1,158 | Random subset, seed 42, of the full release (7,491 authentic, 5,123 tampered) |
| CASIA v1.0 | External test | 1,721 | 800 | 921 | Complete original release |
| CoMoFoD | External test | 970 | 485 | 485 | Balanced subset, no added post-processing |
| MICC-F2000 | External test | 2,000 | 1,300 | 700 | Complete original release |

### CASIA v2.0: leakage-aware partition

In CASIA v2.0, tampered images are derived from authentic images in the same collection, so a random image-level split can leak source content into the test set. The split here works as follows:

1. Remove exact duplicates by SHA-256 hash: 2 authentic images removed, leaving 2,327 (1,169 authentic, 1,158 tampered).
2. Extract source IDs from file names and compute perceptual hashes (Hamming distance <= 4). This found 398 near-duplicate pairs (64 authentic–tampered, 334 tampered–tampered).
3. Merge dependent images into connected groups: 1,290 groups, 1,173 singletons, and a largest group of 828 images (35.6%).
4. Partition the groups with seed 42 and exclude the 209 images (1 authentic, 208 tampered) whose sources fell in different subsets.

| Split | Authentic | Tampered | Total |
|---|---|---|---|
| Train | 747 | 595 | 1,342 |
| Validation | 187 | 140 | 327 |
| Test | 234 | 215 | 449 |
| **Total retained** | **1,168** | **950** | **2,118** |


---

## Method

**Compression Error Analysis (CEA).** Each image is recompressed as JPEG at quality 90. The pixel-wise absolute difference between the original and the recompressed image is scaled to the full 0–255 range and resized to 160 x 160 x 3. CEA follows the recompression principle of Error Level Analysis (ELA).

**Feature extraction.** ConvNeXt-Tiny (ImageNet weights) turns the CEA map into a 5 x 5 x 768 feature map.

**Refinement.** A single ResNeSt split-attention block expands the channels (768 -> 1536), applies a 3x3 grouped convolution with radix-2 split attention (attention bottleneck of 384 units), projects back to 768 channels and adds an identity shortcut. The attention is channel-wise.

**Classification head.** Global average pooling, dropout 0.5, one dense unit with a sigmoid output. An image is labeled tampered if p >= 0.5.

**Training setup**

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras, Google Colab (NVIDIA Tesla T4) |
| Input size | 160 x 160 x 3 |
| Optimizer | Adam |
| Stage 1 (frozen backbone) | 25 epochs, learning rate 1e-4 |
| Stage 2 (full fine-tuning) | 100 epochs, learning rate 1e-5 |
| Scheduler | ReduceLROnPlateau (monitor val loss, factor 0.5, patience 3, min LR 1e-7) |
| Batch size | 32 |
| Loss | Binary cross-entropy with logits, class weights 0.898 (authentic) / 1.128 (tampered) |
| Augmentation (train only) | Horizontal flip, rotation about +/-36 degrees, zoom +/-10% |
| Random seed | 42 |
| Evaluation weights | Final fine-tuning epoch (no checkpoint selection on the test set) |

---

## Results

### Held-out CASIA v2.0 test set (449 images)

| Metric | Value|
|---|---|
| Accuracy | 97.33%|
| Precision (weighted) | 97.34% |
| Recall (weighted) | 97.33%|
| F1-score (weighted) | 97.33% |
| ROC-AUC | 0.986 |

Class-wise results:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Authentic | 96.64% | 98.29% | 97.46% |
| Tampered | 98.10% | 96.28% | 97.18% |

