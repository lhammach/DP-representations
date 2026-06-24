# DP-representations

This repository aims at comparing the internal representations of Neural Networks trained on image classification with Differential Privacy.
The NN used are ResNet18, and they are trained on CIFAR10. 

The notebook `dpsgd_baseline.ipynb` trains networks and saves them in a ./networks folder. The training with DPSGD followed the Opacus tutorial available here : https://opacus.ai/tutorials/building_image_classifier. 

The notebook `cka_analysis.ipynb` computes the CKA scores (centered kernel alignment) for baselines and private networks and plots different comparisons.

# Notebooks outlines

## `dpsgd_baseline.ipynb`

### 0. Setup
### 1. Hyperparameters
### 2. Dataset
### 3. Models
#### 3.1 Building a DP-compatible ResNet18
#### 3.2 Instantiating baseline and DP networks
### 4. Training
#### 4.1 Training the baseline
#### 4.2 Training the DPSGD network


## `cka_analysis.ipynb`
### 0. Setup


# Methodology

Compared scenarios :
- ResNet with SGD 
- ResNet with DP-SGD (clipping + Gaussian noise) for different epsilon values

We keep track of the privacy accountant for each NN with the Rényi divergence.

We measure the similarity between layers of different NNs with the centered kernel alignment (CKA).

We use different seeds to have a confidence interval for the CKA score.

