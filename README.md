# DP-representations

This repository aims at comparing the internal representations of Neural Networks trained on image classification with Differential Privacy.
The NN used are ResNet8, and they are trained on CIFAR10. We follow the Opacus tutorial available here : https://opacus.ai/tutorials/building_image_classifier.

Compared scenarios :
- ResNet with SGD
- ResNet with DP-SGD (clipping + Gaussian noise) for different epsilon values

We keep track of the privacy accountant for each NN with the Rényi divergence.

We measure the similarity between layers of different NN with the centered kernel alignment (CKA).

We use different seeds to have a confidence interval for the CKA score.