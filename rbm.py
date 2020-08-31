import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import cv2

class RBM():

    # hidden = latent space layer
    # observer = input
    def __init__(self, n_hidden=2, m_observe=784, alpha=0.01):
        
        self.n_hidden = n_hidden
        self.m_visible = m_observe

        # initialize layers and biases
        self.visible = None
        self.weight = np.random.rand(self.m_visible, self.n_hidden)
        self.a = np.random.rand(self.m_visible, 1)
        self.b = np.random.rand(self.n_hidden, 1)

        self.alpha = alpha

    
    def sigmoid(self, z):
        return 1 / (1  + np.exp(-z))

    def train(self, data, epochs=2):
        self.visible = data.reshape(-1, self.m_visible)
        self.contrastive_divergence(self.visible, epochs)

    # computing latent variable distribution
    def visible_to_hidden(self, v):
        h_dist = self.sigmoid(
            np.matmul(self.weight.T, v) + self.b)  # [n, 1]
        
        return self.sampling(h_dist)

    # computing visible layer distribution
    def hidden_to_visible(self, h):
        h_dist = self.sigmoid(np.matmul(self.weight, h) + self.a)
        return self.sampling(h_dist)    

    # where temperature takes place
    # T assumed to be 1
    def sampling(self, distribution):
        dimensions = distribution.shape[0]
        true_idx = np.random.uniform(0,1,dimensions).reshape(dimensions, 1) <= distribution
        sampled = np.zeros((dimensions,1))
        sampled[true_idx] = 1
        return sampled

    # contrastive divergence
    def CD_1(self, v_n):
        v_n = v_n.reshape(-1, 1)
        
        # Daydream phase
        h_sampled = self.visible_to_hidden(v_n)

        # reality (negative) phase
        # sample visible from hidden space
        v_sampled = self.hidden_to_visible(h_sampled)
        h_recon = self.visible_to_hidden(v_sampled)

        # change in weight = expected fixed visible - expected sampled visible
        self.weight += self.alpha * v_n * h_sampled.T - v_sampled * h_recon.T
        
        self.a += self.alpha * (v_n - v_sampled)
        self.b += self.alpha * (h_sampled - h_recon)

        # update energy list
        self.energy_list.append(self.calculate_energy(v_n, h_recon))

    # training
    def contrastive_divergence(self, data, max_epoch):
        for _ in range(max_epoch):
            
            np.random.shuffle(data)
            self.energy_list = []

            # Multithreading for CD training, doesn't affect training process overall
            pool = ThreadPool(5)
            pool.map(self.CD_1, data)

            avg_energy = np.mean(self.energy_list)
            print("average energy={}".format(avg_energy))

    # Energy function 
    def calculate_energy(self, visible, hidden):
        return - np.inner(self.a.flatten(), visible.flatten()) - np.inner(self.b.flatten(), hidden.flatten()) \
            - np.matmul(np.matmul(visible.Y, self.weight), hidden)

    def energy(self, v):
        hidden = self.visible_to_hidden(v)
        return self.calculate_energy(v, hidden)

    def Gibbs_sampling(self, v_init, num_iter):
        v_t = v_init.reshape(-1, 1)
        for _ in range(num_iter):
            h_dist = self.sigmoid(np.matmul(self.weight.T, v_t) + self.b)

            h_t = self.sampling(h_dist)

            v_dist = self.sigmoid(np.matmul(self.weight,h_t) + self.a)
            
            v_t = self.sampling(v_dist)

        return v_t, h_t


    def sample(self, iter, v_init):
        v, _ = self.Gibbs_sampling(v_init, iter)
        return v


# for transforming the input data to binary data for RBM visible layer
def threshold(data):

    # make sure the data type is 8 or 16 for the source array 
    img = np.array(data, dtype = np.uint8)

    # Apply OTSU thresholding to transform the input
    ret, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

# Data extraction
def load_data(num_samples = 100):
    # 1059
    training_set = 'empty'

    # randomly choose subset
    random_set = np.random.choice(1058, num_samples, replace=False) + 1

    for sample in random_set:
        
        # threshold and read each image
        training_sample = threshold(np.array(cv2.imread("../data/africa_fabric/africa_fabric/{sample_no:04d}.jpg".format(sample_no=sample),0)).reshape(-1, 1))
        if training_set == 'empty':
            training_set = training_sample
        else:
            training_set = np.append(training_set, training_sample, 1)
	
    return np.array(training_set)

num_hidden = 1000

# data set works best with low numbers
num_samples = 4
running = input("Press 'q' to quit, any other buttn to continue : ")
while running != 'q':
    dataset = load_data(num_samples)
    rbm = RBM(num_hidden, 64 * 64)

    print("Start RBM training")
    rbm.train(dataset,epochs=100)
    print("finished training")

    v = rbm.sample(400, v_init=dataset.T[np.random.randint(num_samples)])
    plt.imshow(v.reshape((64,64)), cmap='gray', aspect='auto')
    plt.axis('off')
    plt.show()
    running = input("Press 'q' to quit, any other buttn to continue : ")