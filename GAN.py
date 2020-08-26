from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import keras.backend as K
import queue

'''
0 Tshirt/top
1 Trouser
2 Pullover/Jumper
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle Boot
'''

# hyper parameters and training variables
batch_size = 4000
epochs = 100

# variables for stability tuning
latent_dimensions = 100
learning_rate = 0.0002
adam_beta = 0.5

# Data extraction
def load_data():
	training_set = np.array(pd.read_csv("./data/fashion-mnist_train.csv"))
	training_samples = training_set[:,1:]
	training_labels = training_set[:,0]

	testing_set = np.array(pd.read_csv("./data/fashion-mnist_test.csv"))
	testing_samples = testing_set[:,1:]
	testing_labels = testing_set[:,0]

	real_set = np.append(training_samples, testing_samples, 0)
	real_labels = np.append(training_labels, testing_labels, 0)

	selected_ix = real_labels == 9
	selected_o_ix = real_labels == 7
	real_set_1 = real_set[selected_ix]
	real_set_2 = real_set[selected_o_ix]
	real_set = np.append(real_set_2, real_set_1, 0)
	real_set = real_set.astype('float32')
	real_samples = (real_set - 127.5) / 127.5

	real_samples = np.reshape(real_samples,(real_samples.shape[0],28,28))
	real_samples = np.expand_dims(real_samples,-1)
	return real_samples


def generate_noise(latent_dim, n_samples):
	# generate points in the latent space
	noise = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	noise = noise.reshape(n_samples, latent_dim)
	return noise

def generate_fake_samples(generator, latent_dim, n_samples):
	
	# generate points in latent space
	x_input = generate_noise(latent_dim, n_samples)
	
	# put noise throught generator network
	X = generator.predict(x_input)

	# create labels as false
	y = np.zeros((n_samples, 1))
	return X, y

def generate_real_samples(dataset, n_samples):

	# select random subsample from real dataset
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]

	# generate class labels
	y = np.ones((n_samples, 1))
	return X, y

def define_discriminator_CNN(in_shape=(28,28,1)):
	# weight initialization
	init = keras.initializers.RandomNormal(stddev=0.02)
	# define model
	model = keras.Sequential()
	# downsample to 14x14
	model.add(keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	# downsample to 7x7
	model.add(keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	# classifier
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	# compile model
	opt = keras.optimizers.Adam(lr=learning_rate, beta_1=adam_beta)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator_CNN(latent_dim):
	# weight initialization
	init = keras.initializers.RandomNormal(stddev=0.02)
	# define model
	model = keras.Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(keras.layers.Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	# output 28x28x1
	model.add(keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
	return model

def define_discriminator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, activation="tanh"))
    model.add(keras.layers.Dense(200, activation="tanh"))
    model.add(keras.layers.Dense(300, activation="tanh"))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=adam_beta),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
    )

    return model

def define_generator():
	# does not need to be compiled as will not directly be trained
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(784, activation='sigmoid'))

    return model

def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = keras.Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta))
	return model

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	
	# plot images
	for i in range(10 * 10):
		plt.subplot(10, 10, 1 + i)
		
		plt.imshow(X[i,:].reshape(28,28), cmap='gray_r')
	
	# save plot to file
	plt.savefig('results_collapse/generated_plot_%03d.png' % (step+1))
	plt.close()
	
	# save the generator model
	g_model.save('results_collapse/model_%03d.h5' % (step+1))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	plt.subplot(2, 1, 1)
	plt.plot(d1_hist, label='discriminator-real-loss')
	plt.plot(d2_hist, label='discriminator-real-loss')
	plt.plot(g_hist, label='generator-loss')
	plt.legend()
	# plot discriminator accuracy
	plt.subplot(2, 1, 2)
	plt.plot(a1_hist, label='real-accuracy')
	plt.plot(a2_hist, label='fake-accuracy')
	plt.legend()
	# save plot to file
	plt.savefig('results_collapse/plot_line_plot_loss.png')
	plt.close()

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
	# calculate the number of batches per epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
	
	# calculate the total iterations based on batch and epoch
    n_steps = bat_per_epo * n_epochs
	
	# calculate the number of samples in half a batch
    half_batch = int(n_batch / 2)
	
	# prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()

	# fail state queue for detecting failure to converge
    fail_conv = queue.Queue(20)
	
    for i in range(n_steps):
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

		# stacking input to reduce failure to converge
        X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
        
        d_loss1, d_acc1 = d_model.evaluate(X_real, y_real)
		# update discriminator model weights
        d_loss2, d_acc2 = d_model.evaluate(X_fake, y_fake)

        _, _ = d_model.train_on_batch(X, y)
        

        X_gan = generate_noise(latent_dim, n_batch)
		# create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
		# update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
		
		# fail state update
        fail_conv.put_nowait((d_loss1, g_loss))

		# summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
            (i+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
		# record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
		# evaluate the model performance every 'epoch'
        if (i+1) % bat_per_epo == 0:
            summarize_performance(i, g_model, latent_dim)

			# check for convergence failures
            if fail_conv.full():
                counter = 0
                for i in range(fail_conv.qsize()):
                    pair = fail_conv.get()
                    if pair[0] == 0 and pair[1] > 10:
                        counter += 1
                        if counter > fail_conv.maxsize/2:
                            print("Failure to converge detected ")
                            print("Ending training ...")
                            return
				
    plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)


# create the discriminator
discriminator = define_discriminator_CNN()
# create the generator
generator = define_generator_CNN(latent_dimensions)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_data()
# train model
train(generator, discriminator, gan_model, dataset, latent_dimensions, 25)