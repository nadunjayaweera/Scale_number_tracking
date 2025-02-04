import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = "image/8.JPG"
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    print(f"Error: Could not read the image at path: {image_path}")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray = cv2.equalizeHist(gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Apply morphological operations to clean up noise and enhance digits
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'

# Perform OCR on the full processed image
extracted_text = pytesseract.image_to_string(morph, config=custom_config)


extracted_numbers = ''.join(filter(lambda x: x.isdigit() or x == '.', extracted_text))

# Display processed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray'), plt.title("Grayscale Image")
plt.subplot(1, 3, 2), plt.imshow(thresh, cmap='gray'), plt.title("Thresholded Image")
plt.subplot(1, 3, 3), plt.imshow(morph, cmap='gray'), plt.title("Processed for OCR")
plt.show()

# Print the extracted weight result
print("Extracted Weight:", extracted_numbers)
