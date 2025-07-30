# Adversarial-Attack-MNIST
Adversarial attack and defense of ML algorithms 
.This phase evaluates how a CNN trained on MNIST behaves under adversarial attacks and how two defense strategies—adversarial training and defensive distillation—impact robustness. The work complements FYP‑I (NIDS), but this README focuses only on the MNIST experiments. 

What we built
A 2D CNN with two sequential conv blocks and three fully‑connected layers; trained with Adam and NLLLoss. 

MNIST dataset: 60,000 training images and 10,000 test images (grayscale 28×28). 

Attacks
FGSM (Fast Gradient Sign Method): uses the sign of the input gradient to craft perturbations that maximize the loss. 

BIM (Basic Iterative Method): iterative variant that repeatedly applies small FGSM steps until an ε‑bound is reached. 

Targeted FGSM: drives the model toward a specific (incorrect) target class by optimizing the loss for that class. 

Defenses
Adversarial Training: augments training with FGSM‑crafted examples to improve robustness. 

Defensive Distillation: trains a student network on soft labels (class probabilities) from a teacher model. 

What we ran & visualized
Attacks evaluated on:

a normal CNN,

an adversarially trained CNN, and

a distilled CNN.
The slide deck includes figures for FGSM/BIM on each model and training/validation loss curves for normal vs. defense models. 

High‑level takeaways
The pipeline demonstrates how small, gradient‑based perturbations can flip predictions on MNIST and how training with adversarial examples or distillation can mitigate some of that vulnerability. 
