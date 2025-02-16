import cv2
import json
import numpy as np
import mysql.connector
from mysql.connector import Error
from pyzbar.pyzbar import decode
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load labels for ImageNet classes
with open('c:/IERG4998/imagenet_labels.json') as f:
    labels = json.load(f)

# Load the trained model for image classification
model = ResNet50(weights='imagenet')  # Load the ResNet50 model with pre-trained ImageNet weights

def get_db_connection():
    """Establish a connection to the MySQL database."""
    try:
        return mysql.connector.connect(
            host='localhost',  # Update with your database host
            user='root',
            password='a20031021',
            database='my_flask_app'  # Update with your database name
        )
    except Error as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def lookup_product_by_barcode(barcode):
    """Lookup product information based on barcode."""
    with get_db_connection() as db:  # Using context manager for automatic closure
        if db is None:
            logging.error("Database connection failed.")
            return None

        cursor = db.cursor(dictionary=True)
        try:
            logging.debug(f"Looking up barcode: {barcode}")
            cursor.execute("SELECT * FROM Products WHERE barcode = %s", (barcode,))
            product = cursor.fetchone()
            logging.debug(f"Product found: {product}")
            return product
        except Error as e:
            logging.error(f"Error fetching product: {e}")
            return None

def recognize_barcode(image_path):
    """Recognize barcodes in the given image and return product information."""
    logging.info(f"Checking image_path: {image_path}")

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        logging.error("Error: Could not read the image.")
        return []

    # Process the image for barcode detection
    logging.debug("Processing image for barcode detection.")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

    # Save intermediate images for debugging
    cv2.imwrite("uploads/gray_image.jpg", gray_image)
    cv2.imwrite("uploads/blurred_image.jpg", blurred_image)
    cv2.imwrite("uploads/thresh_image.jpg", thresh_image)
    cv2.imwrite("uploads/cleaned_image.jpg", cleaned_image)

    # Decode the barcode(s) in the image
    barcodes = decode(image)
    results = []

    # Process detected barcodes
    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        logging.info(f"Decoded barcode: {barcode_data}")
        product_info = lookup_product_by_barcode(barcode_data)

        if product_info:
            results.append({
                'barcode': barcode_data,
                'name': product_info['name'],
                'brand': product_info['brand'],
                'category': product_info['category'],
                'price': product_info['price'],
            })
        else:
            logging.warning(f"Product not found for barcode: {barcode_data}")
            results.append({
                'barcode': barcode_data,
                'name': 'Not found',
                'brand': 'N/A',
                'category': 'N/A',
                'price': 'N/A',
            })

    if not results:  # If no barcodes were found, proceed to image recognition
        logging.info("No barcodes found in the image, attempting item recognition.")
        results = recognize_item_without_barcode(image_path)

    return results

def recognize_item_without_barcode(image_path):
    """Use image recognition to identify an item without a barcode."""
    logging.info("Starting item recognition without barcode.")
    
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))  # Resize for ResNet50
        input_array = keras_image.img_to_array(img)  # Convert to array
        input_array = np.expand_dims(input_array, axis=0)  # Create a mini-batch
        input_array = preprocess_input(input_array)  # Preprocess for ResNet50

        # Make prediction
        predictions = model.predict(input_array)
        
        # Decode predictions
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
        predicted_label = decoded_predictions[0][1]  # Get the label of the highest prediction

        # Lookup product information by name
        category_info = lookup_product_by_name(predicted_label)
        
        logging.info("Finished item recognition.")
        return category_info if category_info else [{'name': predicted_label, 'brand': 'N/A', 'category': 'N/A', 'price': 'N/A'}]
    except Exception as e:
        logging.error(f"Error during item recognition: {e}")
        return [{'name': 'Error during recognition', 'brand': 'N/A', 'category': 'N/A', 'price': 'N/A'}]

def lookup_product_by_name(name):
    """Lookup product information based on the item name."""
    with get_db_connection() as db:  # Using context manager for automatic closure
        if db is None:
            logging.error("Database connection failed.")
            return None

        cursor = db.cursor(dictionary=True)

        try:
            logging.debug(f"Looking up product by name: {name}")
            cursor.execute("SELECT * FROM Products WHERE name LIKE %s", ('%' + name + '%',))
            product = cursor.fetchone()
            logging.debug(f"Product found by name: {product}")
            return product
        except Error as e:
            logging.error(f"Error fetching product by name: {e}")
            return None

# Example usage
if __name__ == "__main__":
    image_path = 'G:\\My Drive\\IERG4998\\captured_image.jpg'  # Replace with the actual image path
    results = recognize_barcode(image_path)
    logging.info(f"Recognition results: {results}")