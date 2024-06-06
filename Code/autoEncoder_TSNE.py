from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

encoding_dim = 16 
input_img = Input(shape=(784,))
# encoded representation of input
encoded = Dense(32, activation="relu")(input_img)
encoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(encoded)
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded representation of code 
decoded = Dense(784, activation='sigmoid')(encoded)
# Model which take input image and shows decoded images
autoencoder = Model(input_img, decoded)

# This model shows encoded images
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
# decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                validation_data=(x_test, x_test))

tsne = TSNE(n_components=2, random_state=42)

class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)
'''plt.figure(figsize=(20, 4))
for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(encoded_img[i].reshape(4, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()'''
tsne_representation = tsne.fit_transform(encoded_img)
for i in range(10):
    class_indices = y_test == i
    plt.scatter(tsne_representation[class_indices, 0], tsne_representation[class_indices, 1], label=class_labels[i], alpha=0.7)
#plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=y_test, cmap=plt.cm.get_cmap("jet", 10))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('t-SNE Visualization of Encoded Test Dataset')
plt.subplots_adjust(right=0.7)
plt.show()