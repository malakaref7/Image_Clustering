import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # resize the image to a fixed size for consistency
    resized_image = cv2.resize(image, (100, 100))
    # convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # threshold the image to create a binary image
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # flatten the binary image into a 1D array
    flattened_image = threshold_image.flatten()
    return flattened_image

def initialize_centroids(k, data):
    # randomly initialize k centroids from the data points
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    return centroids

def assign_labels(data, centroids):
    # assign labels to data points based on the closest centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    return labels

def update_centroids(data, labels, k):
    # Update centroids based on the mean of the assigned data points
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[labels == i], axis=0)
    return centroids


def k_means_clustering(data, k, max_iterations=1000):
    centroids = initialize_centroids(k, data)
    for _ in range(max_iterations):
        labels = assign_labels(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids


def main():
    # Path to the folder containing your training dataset images
    folder_path = 'pattern'
    # Load images from the folder and store them in a list
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, filename))
            if image is not None:
                images.append(image)

    # Preprocess the images to extract color and shape features
    preprocessed_images = [preprocess_image(image) for image in images]
    data = np.array(preprocessed_images)
    # Set the number of clusters
    k = 9
    # Perform k-means clustering
    labels, centroids = k_means_clustering(data, k)

    # Print the number of images in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} images")

    for label in range(k):
        cluster_images = [image for image, cluster_label in zip(images, labels) if cluster_label == label]
        num_columns = len(cluster_images)
        if num_columns > 0:
            fig, axes = plt.subplots(nrows=1, ncols=len(cluster_images), figsize=(10, 10))
            axes = np.ravel(axes)  # use np.ravel() to flatten the axes array
            for i, image in enumerate(cluster_images):
                axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[i].axis("off")
                axes[i].set_title(f"Image {i + 1}")
            plt.suptitle(f"Cluster {label}")
            plt.show()
        else:
            # Handle the case when the number of columns is zero
            print("Number of columns must be a positive integer.")

if __name__ == '__main__':
    main()