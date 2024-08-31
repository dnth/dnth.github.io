from PIL import Image
import numpy as np

def change_color(input_path, output_path, target_color):
    # Open the image
    img = Image.open(input_path).convert('RGBA')
    data = np.array(img)

    # Extract RGB channels
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

    # Create a mask for non-white pixels
    mask = (r != 255) | (g != 255) | (b != 255)

    # Change the color of non-white pixels
    data[:,:,:3][mask] = target_color

    # Create a new image with the modified data
    new_img = Image.fromarray(data)
    
    # Save the new image
    new_img.save(output_path)

# Usage
input_path = 'logo.png'  # Replace with your input image path
output_path = 'output_image.png'  # Replace with your desired output image path
target_color = (102, 119, 249)  # RGB for #6677f9

change_color(input_path, output_path, target_color)
