# Generative Models & Sequence Modeling

## Report preview
<p align="center">
  <img src="assets/Screenshot%20(550).png" alt="Report cutout 1" width="30%" />
  <img src="assets/Screenshot%20(551).png" alt="Report cutout 2" width="30%" />
  <img src="assets/Screenshot%20(552).png" alt="Report cutout 3" width="30%" />
</p>

This project explores image generation with VAEs/GANs and sequence-to-sequence learning for arithmetic
queries. It includes text↔image mappings built with recurrent encoder–decoder models and qualitative
visualizations of generated outputs.

## What’s inside
- `VAE_filter64_latent32.py` / `VAE_filter64_latent64.py` / `VAE_filter64_latent128.py`:
  VAE experiments with different latent sizes.
- `GAN_filtter128_latent64.py` / `GAN_filtter128_latent128.py` / `GAN_filter128_latent256.py`:
  GAN experiments with different latent sizes.
- `A2_RNNs_text2image.py`: text-to-image sequence model for MNIST-style digits.
- `task2_txt2txtimg2txt.py`: text-to-text and image-to-text sequence models.

## Datasets
- CelebA (image generation with VAE/GAN)
- MNIST (digit images for sequence tasks)
- Synthetic arithmetic expressions (text inputs/outputs)

## Tasks covered
- Generative models: train VAE and GAN on a custom image dataset and generate samples.
- Latent interpolation: create smooth transitions between two random latent points.
- Text-to-text: predict arithmetic results from text expressions with varying train/test splits.
- Image-to-text: read digit images + operator and output the result in text.
- Text-to-image: generate MNIST-style digit images for the answer sequence.
- Model variations: explore changes in latent size and recurrent depth.

## Results (high level)
- VAE: smaller latent sizes produced sharper faces, while larger sizes reduced loss but blurred outputs.
- GAN: output sharpness varied less across latent sizes; longer training likely needed for clarity.
- Text-to-text: strong generalization with high accuracy even with limited training data.
- Image-to-text: lower accuracy than text-to-text due to higher input dimensionality.
- Text-to-image: outputs were visually plausible but lacked crisp digit structure.

## Report cutouts (results graphs)
![VAE latent dim 64 (epoch 0/49)](assets/Screenshot%20(548).png)
![GAN latent dim 64 (epoch 0/49)](assets/Screenshot%20(549).png)

## Requirements
- Python 3.x
- `tensorflow` / `tf_keras`
- `numpy`, `matplotlib`, `tqdm`
- `opencv-python`, `scikit-learn`, `scipy`

Install:

```
pip install tensorflow tf_keras numpy matplotlib tqdm opencv-python scikit-learn scipy
```

## Run
These files are exported from notebooks and include cell markers and `%pip` lines. The smoothest way is
to open them in Jupyter/Colab. If you run as plain Python, comment out the `%pip` lines first.

Generative models:

```
python VAE_filter64_latent32.py
python VAE_filter64_latent64.py
python VAE_filter64_latent128.py
python GAN_filtter128_latent64.py
python GAN_filtter128_latent128.py
python GAN_filter128_latent256.py
```

Sequence models:

```
python task2_txt2txtimg2txt.py
python A2_RNNs_text2image.py
```

## Notes
- Training and generation can take time depending on hardware.
- You can adjust latent sizes, filters, and epochs in the scripts to explore trade-offs.
