import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt

name = sys.argv[1]
data = pickle.load(open("multi_mnist_data/common"+name+".pkl", "rb"))
print(np.shape(data['image']))
img = data['image'][10]
plt.imshow(np.squeeze(img))
plt.savefig("test.png")
