# In this code I want to scale the pixel values of the depth image to the range of 0-255

import cv2
import numpy as np

def scale_depth_image(depth_image_path, output_image_path):
    # Read the depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Normalize the pixel values to the range [0, 1]
    depth_min = np.min(depth_image)
    depth_max = np.max(depth_image)
    normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)

    # Scale the normalized values to the range [0, 255]
    scaled_depth = (normalized_depth * 100)

    # Save the scaled depth image
    cv2.imwrite(output_image_path, scaled_depth)

# Example usage

# depth_image_path = 'path/to/depth_image.png'
# output_image_path = 'path/to/scaled_depth_image.png'
# scale_depth_image(depth_image_path, output_image_path)


def test_batch():
    import os
    import glob
    depth_image_dir = '/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/1101_dataset/gimbal/1101_test_3_2/depth_image/'
    output_dir = '/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/scaled_for_paper/depth_scaled/'
    
    # /home/lg1/peteryu_workspace/m2g_concept_graph/dataset/1101_dataset/gimbal/1101_test_3_2/depth_image/YHY7BK_d_0.png
    
    image_start = 0
    iamge_end = 11
    
    for i in range(image_start, iamge_end):
        depth_image_path = depth_image_dir + 'YHY7BK_d_' + str(i) + '.png'
        output_image_path = output_dir + 'YHY7BK_d_' + str(i) + '.png'
        scale_depth_image(depth_image_path, output_image_path)
        print('Scaled depth image saved at:', output_image_path)
        
if __name__ == '__main__':
    test_batch()
