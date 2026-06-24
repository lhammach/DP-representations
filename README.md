# DP-representations

This repository aims at comparing the internal representations of Neural Networks trained on image classification with Differential Privacy.
The NN used are ResNet18, and they are trained on CIFAR10. 

The notebook `dpsgd_baseline.ipynb` trains networks and saves them in a `./networks` directory. The DP-SGD training followed the [Opacus tutorial](https://opacus.ai/tutorials/building_image_classifier). 

The notebook `cka_analysis.ipynb` computes the CKA (Centered Kernel Alignment) scores for both baselines and private networks, and plots different comparisons.

# Notebooks outlines

## `dpsgd_baseline.ipynb`

### 0. Setup
### 1. Hyperparameters
### 2. Dataset
### 3. Models
#### 3.1 Build a DP-compatible ResNet18
#### 3.2 Instante baseline and DP networks
### 4. Training
#### 4.1 Train the baseline
#### 4.2 Train the DPSGD network


## `cka_analysis.ipynb`
### 0. Setup
### 1. Dataset
#### 1.1 Data preprocessing
#### 1.2 Evaluation dataloader
### 2. Rebuild models
#### 2.1 Corrections for Opacus compatibility
#### 2.2 Build baseline and DP
#### 2.3 Load checkpoints
#### 2.4 Accuracy plots
### 3. Extract intermediate representations
### 4. CKA
#### 4.1 Formulas
#### 4.2 (à modifier) run on several batches
### 5. (à modifier) CKA
### 6. (à modifier) Plots



# Methodology

Compared scenarios :
- ResNet with SGD 
- ResNet with DP-SGD (clipping + Gaussian noise) for different epsilon values

We keep track of the privacy accountant for each NN with the Rényi divergence.

We measure the similarity between layers of different NNs with the centered kernel alignment (CKA).

We use different seeds to have a confidence interval for the CKA score.

