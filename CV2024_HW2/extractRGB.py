import cv2
import numpy as np
import random
import os

def apply_random_offset(channel):
    """
    Apply a small random offset to the channel.
    """
    rows, cols = channel.shape
    # Generate random offsets for x and y within a range
    offset_x = random.randint(-10, 10)
    offset_y = random.randint(-10, 10)

    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    
    # Apply affine transformation
    offset_channel = cv2.warpAffine(channel, translation_matrix, (cols, rows))
    
    return offset_channel

def main():
    # Load the image
    image_path = 'CV2024_HW2/my_data/custom.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # Split the image into its BGR channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply random offsets to each channel
    r_offset = apply_random_offset(r_channel)
    g_offset = apply_random_offset(g_channel)
    b_offset = apply_random_offset(b_channel)

    # Ensure the output directory exists
    output_dir = 'CV2024_HW3/output'
    os.makedirs(output_dir, exist_ok=True)

    # Save the offset images
    cv2.imwrite(os.path.join(output_dir, 'red_channel_offset.jpg'), r_offset)
    cv2.imwrite(os.path.join(output_dir, 'green_channel_offset.jpg'), g_offset)
    cv2.imwrite(os.path.join(output_dir, 'blue_channel_offset.jpg'), b_offset)

    print(f"Offset images saved in {output_dir}")

    # Optionally display the offset images
    cv2.imshow('Red Channel (Offset)', r_offset)
    cv2.imshow('Green Channel (Offset)', g_offset)
    cv2.imshow('Blue Channel (Offset)', b_offset)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
