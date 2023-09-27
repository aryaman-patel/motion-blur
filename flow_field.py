# Generate the flowfield from .npy files
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.preprocessing import LabelEncoder


def inverse_transform_handling_unknowns(encoder, values):
    try:
        return encoder.inverse_transform(values)
    except ValueError as e:
        # round off unknown values to the nearest value
        unknowns = set(values) - set(encoder.classes_)
        for unknown in unknowns:
            nearest_value = min(encoder.classes_, key=lambda x: abs(x - unknown))
            values[values == unknown] = nearest_value
        return encoder.inverse_transform(values)


def generate_flowfield(image_path, prediction_path):
    """
    Function to generate the flowfield from .npy files
    """
    le_u = LabelEncoder()
    le_v = LabelEncoder()
    le_u.classes_ = np.loadtxt("/home/patel.aryam/DeBlur/labels_u.txt")
    le_v.classes_ = np.loadtxt("/home/patel.aryam/DeBlur/labels_v.txt")
    # Read the prediction file
    prediction = np.load(prediction_path)
    # Read the image file
    image = mpimg.imread(image_path)

    print("Prediction shape:", prediction.shape)

    u_values = prediction[0:64, :, :]
    u_values = u_values.astype(int)
    v_values = prediction[64:128, :, :]
    v_values = v_values.astype(int)
    
    # Function to return U V values - 
    u_values = np.max(u_values, axis=0)
    v_values = np.max(v_values, axis=0)
    
    # u_final = inverse_transform_handling_unknowns(le_u, u_values.flatten()).reshape(u_values.shape)
    # v_final = inverse_transform_handling_unknowns(le_v, v_values.flatten()).reshape(v_values.shape)
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
    plt.savefig('/home/patel.aryam/DeBlur/flowfield.png')

def main() :
    # Generate the flowfield from .npy files
    image_path = '/home/patel.aryam/DeBlur/dataset_old/data_syn/train/8143_012_blurryimg.png'
    prediction_path = '/home/patel.aryam/DeBlur/result_flowfield.npy'
    generate_flowfield(image_path, prediction_path)


if __name__ == "__main__" :
    main()