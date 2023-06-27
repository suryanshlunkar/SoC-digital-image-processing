import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load .mat file
file = h5py.File('data/mnist.mat', 'r')

labels = np.array(file['labels_train']).T
images = np.array(file['digits_train']).T.astype(float)  # Convert to float

# Compute mean and covariance matrix for each digit
means = []
cov_matrices = []
for digit in range(10):
    digit_images = []
    for i in range(len(labels)):
        if labels[i][0] == digit:
            digit_images.append(images[:, :, i])
    digit_images = np.array(digit_images)
    digit_mean = np.mean(digit_images, axis=0)
    digit_cov = np.cov(digit_images.reshape(-1, 28*28), rowvar=False)
    means.append(digit_mean)
    cov_matrices.append(digit_cov)


# Compute eigenvalues and eigenvectors for each covariance matrix
eigenvalues = []
eigenvectors = []

for cov in cov_matrices:
    eigvals, eigvecs = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigvals)[::-1]  # Sort in descending order
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    eigenvalues.append(sorted_eigvals)
    eigenvectors.append(sorted_eigvecs)

# Plot eigenvalues for each digit
plt.figure(figsize=(12, 8))

for digit in range(10):
    plt.plot(eigenvalues[digit], label=str(digit))

plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues for each digit')
plt.legend()
plt.show()

# Display images showing the principal mode of variation
plt.figure(figsize=(16, 6))

for digit in range(10):
    plt.subplot(3, 10, digit+1)


    # Calculate principal mode of variation
    mean_image = means[digit]
    eigenvalue = eigenvalues[digit][0]
    eigenvector = eigenvectors[digit][:, 0]
    variation = np.sqrt(eigenvalue) * eigenvector

    # Reshape variation to match the shape of mean_image
    variation = variation.reshape(28, 28)

    # Create images
    image_1 = mean_image - variation
    image_3 = mean_image + variation

    # Normalize pixel values between 0 and 255
    image_1 = (image_1 - np.min(image_1)) / (np.max(image_1) - np.min(image_1)) * 255
    image_3 = (image_3 - np.min(image_3)) / (np.max(image_3) - np.min(image_3)) * 255

    # Convert to uint8 and create PIL Image objects
    image_1 = Image.fromarray(image_1.astype(np.uint8))
    mean_image = Image.fromarray(mean_image.astype(np.uint8))
    image_3 = Image.fromarray(image_3.astype(np.uint8))

    # Display images
    plt.imshow(image_1, cmap='gray')
    plt.axis('off')
    plt.subplot(3, 10, digit+11)

    plt.imshow(mean_image, cmap='gray')
    plt.axis('off')
    plt.subplot(3, 10, digit+21)
    plt.imshow(image_3, cmap='gray')
    plt.axis('off')

plt.suptitle('Principal Mode of Variation for Each Digit')
plt.show()

