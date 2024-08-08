"""
RaspiCar
Copyright (c) 2024 Fynn Luca Maaß

Licensed under the Custom License. See the LICENSE file in the project root for license terms.
"""

import cv2

def resize_and_crop_image(image, target_width=200, target_height=66):
    # Check input image dimensions
    if len(image.shape) < 2:
        raise ValueError("Invalid image data!")

    height, width = image.shape[:2]

    # Calculate scaling factor to maintain aspect ratio based on width
    scaling_factor = target_width / width
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Check if the new height is greater than or equal to the target height before cropping
    if new_height < target_height:
        raise ValueError("Resized image height is less than the target crop height.")

    # Calculate start y-coordinate for cropping to center the crop area
    y_start = new_height - target_height

    cropped_image = resized_image[y_start:y_start + target_height, 0:target_width]
    # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    return cropped_image