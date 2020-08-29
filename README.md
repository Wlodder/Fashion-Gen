# Fashion-Gen
 Generative modelling of fashion articles

Using DCGANs to generate silhouettes and generic features of designs for shoes based of the Fashion MNIST data set. The model was trained using 
dataset found here: https://www.kaggle.com/zalando-research/fashionmnist. Specifically the samples labelled '7' and '9', Sneaker and Ankle Boot, due to
the easily distinguishable outlines for training. The same model works for 3 sets of data produced where the real data set consisted of:
* Sneaker
* Ankle Boot
* Sneaker and Ankle Boot

Associated loss and accuracy graphs were produced for each of the training sessions. Lack of convergence was a particular problem in training so a fail state
was placed in the training procedure to detect however it will cause the training to be cut short when sometimes the model can recover and produce okay samples. GAN was used for its ability to generate high dimensional data without the restriction of having to use a binary input like a RBM or Hopfield network.

An over lay generated patterns for an RBM trained using https://www.kaggle.com/mikuns/african-fabric and processed using OTSU binary threshold to prepare data for the RBM's
visible layer, each of the images is 64 x 64 pixels. 

## Things to add
* Wasserstein loss function for increased stability of training
* Image processing to sharpen images

## Resources used
### GAN
* https://developers.google.com/machine-learning/gan/training
* https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
* https://arxiv.org/abs/1701.00160

### RBM
* http://deeplearning.net/tutorial/rbm.html
* https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
* https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9 Lectures 11.1 - 12.5 (Hopfield networks and RBMs)
* https://www.youtube.com/watch?v=p4Vh_zMw-HQ&list=PLmvaDFcAzchCIqoHUX_C5tMxRpL0rXcdO
