## Overview
This project demonstrates the power of **Generative Adversarial Networks (GANs)** by generating realistic-looking fashion images based on the Fashion **Modified National Institute of Standards and Technology (MNIST)** dataset. The generator and discriminator are built using **PyTorch** and trained on the **28x28 grayscale** clothing images.

---

## Key Features
**DCGAN Architecture:** Stable GAN training using convolutional layers and batch normalization.
**Real-time Image Generation:** Save generated fashion items every epoch for visual progress tracking.
**PyTorch-based:** Clean, modular, and extensible code using the PyTorch deep learning framework.
**Model Checkpointing:** Save and reload generator/discriminator models.
**Visual Feedback:** Generates generated_images/epoch_*.png at each epoch.

---

## How to Run
**1. Install Requirements**
```
pip install torch torchvision matplotlib
```

**2. Run the GAN Training Script**
```
python fashion_gan.py
```

**3. Output**

Generated images will be saved in 
```
generated_images/.
```

Model weights saved in 
```
models/.
```

---

## Future Enhancements
Add **GAN Loss graphs (Generator vs Discriminator loss)**.
Integrate with **Streamlit** for **interactive generation**.
Try conditional **GANs (cGANs)** for **category-specific image generation**.
Upgrade to **StyleGAN** or **Diffusion models** for **higher realism**.

---

## Contribute
Pull requests are welcome! If you have enhancements, cleaner training loops, or new features (e.g., loss plots or conditional GANs), feel free to fork and submit.

---

## Summary
**GAN-MNIST** lets you learn, experiment, and generate — all in one project. Perfect for anyone exploring GANs or building a practical deep learning portfolio.

---

**Happy Generating! 👠👚👗**
