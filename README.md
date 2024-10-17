# VAE_Sample_Size
# Variational Autoencoder (VAE) for X-ray Image Reconstruction
This project demonstrates the use of a Variational Autoencoder (VAE) to reconstruct X-ray images. The goal is to compare the classification performance of a Random Forest (RF) model on the original images versus the VAE-reconstructed images. This project also explores the effect of varying the sample size on classification performance.

# Project Overview
Variational Autoencoders (VAEs) are powerful generative models that can encode high-dimensional data into a lower-dimensional latent space. This project uses a VAE to compress and reconstruct X-ray images. We then evaluate how well a Random Forest classifier performs on both the original and VAE-reconstructed images with varying amounts of training data.

# Features
Training a VAE on X-ray images.
Reconstruction of images using VAE.
Comparison of classification performance on original vs. reconstructed images.
Exploration of sample size effects on classification accuracy.

# Installation
To run this project, clone the repository and install the required packages:
git clone https://github.com/username/vae-xray-reconstruction.git
cd vae-xray-reconstruction
pip install -r requirements.txt

