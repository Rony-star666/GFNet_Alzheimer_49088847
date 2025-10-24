#  GFNet Alzheimer’s MRI Classifier

## Model and Problem Description 
The GFNet adopts a transformer-inspired architecture that replaces local convolutions with global Fourier-based filtering, allowing efficient long-range dependency modeling.
- **Global Fourier Filtering**: Each layer transforms the feature map to the frequency domain using FFT, applies learnable complex filters, and converts back via inverse FFT. This enables efficient global token mixing across the entire image.
- **Feed-Forward Network (FFN)**: After filtering, a two-layer MLP with GELU activation and dropout refines the features and enhances non-linear representation.
- **Normalization & Residuals**: Layer Normalization is applied before each sub-block with residual connections for stable and efficient training.
- **Projection & Classification Head**: An initial projection layer expands input channels using 1×1 convolution, followed by stacked GF blocks, global average pooling, and a linear classifier for final AD vs. NC prediction.
- **Efficiency**: The FFT-based operation provides global receptive fields with low computational cost, making GFNet suitable for high-resolution 2D MRI images.

## Dataset
- **Source**: Data can be accessed and downloaded via the following link:
https://adni.loni.usc.edu/data-samples/adni-data/
Content: It includes 2D T1-weighted MRI brain slices, each labeled as Alzheimer’s disease (AD) or Normal Control (NC).
- **Purpose**: These MRI slices are used to train a binary classifier that distinguishes AD from NC subjects based on structural brain features.
- **Preprocessing**:
Resized all images to 224×224 pixels.
Normalized grayscale intensity values to [−1, 1].
Applied gamma correction and Gaussian noise for contrast enhancement and data augmentation.
- **Data Split**:
Training, validation, and testing sets are divided on a subject-wise basis to prevent data leakage.
The dataset is approximately class-balanced to ensure unbiased evaluation.
## Usage
### Training
The following are the configuration parameters required for model training:
| Argument | Description | Type | Default |
| ----- | ----- | ----- | ----- |
| `data_root`               | Root directory of the dataset, e.g., .../root/AD_NC | `str` | Required |
| `outdir`         | Output directory for saving checkpoints and plots | `str` | runs/adni_gfnetquired |
| `img_size` | Input image resolution | `int` | 224 |
| `batch_size` | Number of samples per training iteration | `int` | 32 |
| `workers` | Number of data loading workers | `int` | 16 |
| `epochs` | Total number of training epochs | `int` | 80 |
| `lr`| Initial learning rate (with cosine warmup schedule) | `float` | 3e-4 |
| `weight_decay` | L2 regularization strength | `float` | 5e-3 |
| `in_channels` | Number of input channels  | `int` | 1 |
| `seed` | Random seed for reproducibility | `int` | 42 |
- **Learning rate schedule**: Linear warmup (first 3 epochs) followed by cosine decay to 1e-6.
- **Optimizer**: AdamW (decoupled weight decay).
- **Loss function**: Cross-entropy with optional label smoothing (0.02).
- **Regularization**: Dropout (0.3 in model), AutoTuner dynamically adjusts weight decay, label smoothing, and dropout based on training–validation gap.
- **Temperature Scaling**: Applied every 5 epochs using validation set for post-hoc calibration.
- **Early stopping**: Not enabled (patience=0 by default), training proceeds for full epochs.
Use the provided training script (e.g. train.py) to train the model. Example command:
```bash
python train.py --data_root path/to/ADNI/AD_NC --outdir runs/experiment1 \
    --img_size 224 --batch_size 32 --epochs 80 --lr 3e-4 --workers 16 --seed 42
```
### Prediction
| Argument | Description | Type | Default |
| ----- | ----- | ----- | ----- |
| `--ckpt`        | Path to the model weights | `str` | Required |
| `--data_root`   | Root directory of the dataset; must contain subfolders for evaluation  | `str` | Required |
| `--split`       | Subdirectory under data_root for evaluation (e.g., validation, test) | `str` | `validation` |
| `--image`       | Path to a single image for individual inference | `str` | `None` |
| `--batch_size`  | Batch size for folder-based batch prediction | `int` | `64` |
| `--num_workers` | Number of DataLoader workers for inference | `int` | `0` |
| `--device`      | Device for inference (cuda, cpu, or mps) | `str` | `cuda` |
| `--save_dir`    | Directory to save prediction results | `str` | `pred_out` |

### Dependencies
| Dependencies | Version|
| ----- | ----- |
| Python | 3.13.6 | 
| torch | 2.7.1 + cu118 |
| torchvision | 0.22.1 + cu118| 
| numpy| 2.1.2 |
| scikit-learn | 1.7.1 | 
| matplotlib | 3.10.5 | 
| pillow | 11.0.0 | 
| tqdm | 4.67.1 |

## Results
After training on the ADNI MRI slices, the GFNet model achieved strong performance in distinguishing AD vs NC:

Test Accuracy: ~0.81 (81%). This means the model correctly classifies 81% of held-out test images (subject-level evaluation). Given the challenging nature of MRI-based diagnosis, this accuracy demonstrates that the model has learned meaningful biomarkers of Alzheimer’s.

ROC AUC: ~0.85–0.88 on the test set. The Area Under the ROC Curve indicates the model’s discrimination capability between AD and NC across all classification thresholds. An AUC in the high 0.8s suggests the model is capturing the separation between classes well (with an AUC of 1.0 being perfect separation).

The training and validation metrics over epochs are shown in the figures below:
