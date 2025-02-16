# barcode_functions.py
import os  
import cv2 as cv
import logging

from image_processing import process_image
from grocery_capture import GroceryCaptureApp  # Import the Kivy app
from barcode_identification import recognize_barcode

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def capture_image():
    try:
        # Start the Kivy application to capture the image
        GroceryCaptureApp().run()
        return "Image capture process initiated."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def recognize_item(image_path):
    # Call the barcode recognition function with the image path
    results = recognize_barcode(image_path)

    # Prepare a detailed result dictionary
    if results:
        return {
            'image_name': os.path.basename(image_path),  # Get only the filename
            'barcodes': results
            #'message': f"{len(results)} barcode(s) detected." if len(results) > 0 else "No barcodes detected."
            #'message': f"one barcode detected." if len(results) > 0 else "No barcodes detected."
        }
    else:
        return {
            'image_name': os.path.basename(image_path),
            'barcodes': []
             #'message': "No barcodes detected."
        }

def count_quantity_from_image(image_path):
    logging.info(f"Counting quantity from image: {image_path}")

    # Process the image to count objects
    count = process_image(image_path)
    
    if count is not None:
        result_message = f"Counted quantity: {count} items"
        logging.info(result_message)
        return result_message
    else:
        logging.error("Counting failed due to image processing error.")
        return "Error counting items."
