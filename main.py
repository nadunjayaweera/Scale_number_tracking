import cv2
import pytesseract
import numpy as np

# Set Tesseract path (for Windows users, change if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def read_numbers(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image is loaded properly
    if img is None:
        print(f"Error: Could not read the image at path: {image_path}")
        return None  # Return None if the image cannot be loaded

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Tesseract OCR config for digit recognition
    config = "--psm 6 outputbase digits"

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(thresh, config=config)

    # Filter only digits
    numbers = ''.join(filter(str.isdigit, extracted_text))

    return numbers

# Image file path
# image_path = "image/8.JFIF"  # Make sure this is correct!
image_path = "image/8.JPG"
numbers = read_numbers(image_path)

if numbers is not None:
    print("Extracted Numbers:", numbers)
