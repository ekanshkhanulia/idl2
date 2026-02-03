# Generative Models & Sequence Modeling

This project explores generative modeling for images and sequence-to-sequence learning for arithmetic
tasks. It covers VAE/GAN image generation on CelebA and recurrent encoder–decoder models that map
between text, image sequences, and generated digit images.

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

## Results (high level)
- VAE: smaller latent sizes produced sharper faces, while larger sizes reduced loss but blurred outputs.
- GAN: output sharpness varied less across latent sizes; longer training likely needed for clarity.
- Text-to-text: strong generalization with high accuracy even with limited training data.
- Image-to-text: lower accuracy than text-to-text due to higher input dimensionality.
- Text-to-image: outputs were visually plausible but lacked crisp digit structure.

## Report cutouts (results graphs)
Add the cropped result-graph images from your report in `assets/` and update the links below.

![VAE/GAN results](assets/results-generative.png)
![Seq2Seq results](assets/results-sequence.png)

## Run
Install the required Python packages (e.g., TensorFlow/Keras, NumPy, Matplotlib), then run:

```
python VAE_filter64_latent32.py
python GAN_filter128_latent256.py
python task2_txt2txtimg2txt.py
python A2_RNNs_text2image.py
```

## Notes
- Training and generation can take time depending on hardware.
- You can adjust latent sizes, filters, and epochs in the scripts to explore trade-offs.
