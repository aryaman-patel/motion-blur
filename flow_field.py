# Generate the flowfield from mat files
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy.io as sio

def generate_flowfield(image_path, prediction_path):
    """
    Function to generate the flowfield from .npy files
    """
    # Read the prediction file
    prediction = sio.loadmat(prediction_path)
    prediction =  prediction['mfmap']

    # Convert prediction to numpy array
    prediction = np.array(prediction)

    # Read the image file
    image = mpimg.imread(image_path)

    print("Prediction shape:", prediction.shape)

    u_values = np.zeros((300, 460))
    v_values = np.zeros((300, 460))
    u_values[:min(prediction.shape[0], 300), :min(prediction.shape[1], 460)] = prediction[:min(prediction.shape[0], 300), :min(prediction.shape[1], 460), 0]
    v_values[:min(prediction.shape[0], 300), :min(prediction.shape[1], 460)] = prediction[:min(prediction.shape[0], 300), :min(prediction.shape[1], 460), 1]
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a meshgrid for the coordinates
    x = np.arange(0, image.shape[1], 20)
    y = np.arange(0, image.shape[0], 20)
    X, Y = np.meshgrid(x, y)

    # Longer arrows
    ax.quiver(X, Y, u_values[::20, ::20], v_values[::20, ::20])

    # Display the image
    ax.imshow(image)

    # Save the figure
    plt.savefig('DeBlur/flowfield.png')

def main() :
    # Generate the flowfield from .npy files
    image_path = 'DeBlur/dataset_old/data_syn/train/8143_021_blurryimg.png'
    prediction_path = 'DeBlur/result_flowfield.mat'
    generate_flowfield(image_path, prediction_path)


if __name__ == "__main__" :
    main()