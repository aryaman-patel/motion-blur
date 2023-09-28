import os
from PIL import Image

# Check the dimentions of the images.
# directory = 'DeBlur/dataset_old/data_syn/train'  # Replace with the actual directory path

# for filename in os.listdir(directory):
#     if filename.endswith('_blurryimg.png'):
#         filepath = os.path.join(directory, filename)
        
#         # Open the image
#         img = Image.open(filepath)
        
#         # Resize the image to 460x300
#         resized_img = img.resize((460, 300))
        
#         # Save the resized image with the same filename
#         resized_img.save(filepath)

# Convert all the images to the same size. 
directory = 'DeBlur/dataset_old/data_syn/train'  # Replace with the actual directory path

for filename in os.listdir(directory):
    if filename.endswith('_blurryimg.png'):
        filepath = os.path.join(directory, filename)
        size = os.path.getsize(filepath)
        
        image = Image.open(filepath)
        width, height = image.size
        
        print(f'{filename}: Size: {size} bytes, Dimensions: {width} x {height}')
