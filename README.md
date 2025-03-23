# Skin Lesion Segmentation with Swin Transformer + UPerNet

This repository contains an end-to-end deep learning pipeline for semantic segmentation of skin lesions using the **ISIC 2018 Challenge Dataset**. The project leverages the **Swin Transformer** architecture as a hierarchical Vision Transformer encoder, paired with a **UPerNet** decoder for robust multi-scale prediction. This combination enables precise localization of skin lesions, even with ambiguous boundaries.

---

## Highlights

- **Task**: Semantic segmentation of skin lesions (binary masks).
- **Architecture**: Swin Transformer (Tiny) encoder + UPerNet decoder.
- **Domain**: Medical image analysis – dermatology.
- **Frameworks**: PyTorch, Albumentations, Google Colab.

---

## Core Features

- **Vision Transformer Backbone**: Swin Transformer introduces hierarchical self-attention with local windowing and shifted windows for efficient global context modeling.
- **Multi-Scale Decoder**: UPerNet fuses semantic features at different scales, improving boundary delineation.
- **Uncertainty Estimation**: Implemented **Monte Carlo Dropout** to quantify predictive uncertainty during inference.
- **Model Interpretability**: Applied **Grad-CAM** to visualize attention across lesion regions, offering clinical transparency.
- **Baseline Comparison**: UNet-based segmentation implemented for comparative evaluation.
- **Loss Function**: Combined **Dice Loss** and **Binary Cross Entropy (BCE)** for class imbalance handling.
- **Metrics**: Dice Coefficient and IoU.

---

## Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `colab_setup.ipynb` | Google Colab + Drive mounting and initialization. |
| `dataset_preprocessing.ipynb` | Data cleaning, resizing (256×256), and binary mask encoding. |
| `unet_training.ipynb` | Baseline UNet model training for lesion segmentation. |
| `swin_upernet_baseline.ipynb` | Initial Swin + UPerNet implementation with standard training. |
| `swin_upernet_main.ipynb` | Final tuned model with Grad-CAM visualization and MC Dropout. |

---

## Model Weights

The final trained model weights are hosted on Hugging Face:

👉 [Download best_swin_upernet_main.pth](https://huggingface.co/samyakshrestha/swin-medical-segmentation/resolve/main/best_swin_upernet_main.pth?download=true)

To load the model:
import torch
model.load_state_dict(torch.load("best_swin_upernet_main.pth", map_location=torch.device("cpu")))

## Experimental Enhancements

The following were implemented to increase model robustness and interpretability:
	•	Monte Carlo Dropout:
	•	Dropout layers kept active during inference.
	•	Used to estimate epistemic uncertainty by sampling predictions multiple times.
	•	Grad-CAM:
	•	Heatmap visualizations on Swin Transformer outputs.
	•	Highlights the regions that influence predictions most strongly.

## Future Extensions
- Ensembling with Mask2Former
- Test-Time Augmentation (TTA)
- Hausdorff Distance and AUC-ROC as additional evaluation metrics
