# Fashion-Gen
 Generative modelling of fashion articles

Using DCGANs to generate silhouettes and generic features of designs for shoes based of the Fashion MNIST data set. The model was trained using 
dataset found here: https://www.kaggle.com/zalando-research/fashionmnist. Specifically the samples labelled '7' and '9', Sneaker and Ankle Boot, due to
the easily distinguishable outlines for training. The same model works for 3 sets of data produced where the real data set consisted of:
* Sneaker
* Ankle Boot
* Sneaker and Ankle Boot

Associated loss and accuracy graphs were produced for each of the training sessions. Lack of convergence was a particular problem in training so a fail state
was placed in the training procedure to detect however it will cause the training to be cut short when sometimes the model can recover and produce okay samples.

## Things to add
* Wasserstein loss function for increased stability of training
* RBM for pattern generation 
* Image processing to sharpen images

## Resources used
https://developers.google.com/machine-learning/gan/training
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
https://arxiv.org/abs/1701.00160
