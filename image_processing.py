import cv2
import json
import numpy as np
import mysql.connector
from mysql.connector import Error
from pyzbar.pyzbar import decode
import torch
from torchvision import models, transforms
import logging
from flask import Flask
from PIL import Image
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)

# Load labels for ImageNet classes
with open('c:/IERG4998/imagenet_labels.json') as f:
    labels = json.load(f)

# Load the fruit classifier model
fruit_model = torch.load('fruit_classifier_model.pth')
fruit_model.eval()

# Load the general model for object recognition
general_model = models.resnet50(weights='IMAGENET1K_V1')
general_model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_db_connection():
    """Establish a connection to the MySQL database."""
    try:
        return mysql.connector.connect(
            host='localhost',
            user='root',
            password='a20031021',
            database='my_flask_app'
        )
    except Error as e:
        app.logger.error(f"Error connecting to database: {e}")
        return None

def lookup_product_by_barcode(barcode):
    """Lookup product information based on barcode."""
    with get_db_connection() as db:
        if db is None:
            app.logger.error("Database connection failed.")
            return None

        cursor = db.cursor(dictionary=True)
        try:
            app.logger.debug(f"Looking up barcode: {barcode}")
            cursor.execute("SELECT * FROM Products WHERE barcode = %s", (barcode,))
            product = cursor.fetchone()
            app.logger.debug(f"Product found: {product}")
            return product
        except Error as e:
            app.logger.error(f"Error fetching product: {e}")
            return None

def lookup_product_by_name(name):
    """Lookup product information based on the item name."""
    with get_db_connection() as db:
        if db is None:
            app.logger.error("Database connection failed.")
            return None

        cursor = db.cursor(dictionary=True)
        try:
            app.logger.debug(f"Looking up product by name: {name}")
            cursor.execute("SELECT * FROM Products WHERE name LIKE %s", ('%' + name + '%',))
            product = cursor.fetchone()
            app.logger.debug(f"Product found by name: {product}")
            return product
        except Error as e:
            app.logger.error(f"Error fetching product by name: {e}")
            return None

def filter_contours(contours, min_area=1000):
    """Filter contours based on area."""
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # Skip contours that do not meet area criteria

        filtered_contours.append(contour)  # Add the contour if it meets the criteria

    return filtered_contours

def count_objects(original):
    """Count objects in the image using advanced preprocessing and classify them by type."""
    # Convert image to grayscale
    gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply gamma correction
    gray_correct = np.array(255 * (gray_im / 255) ** 1.2, dtype='uint8')

    # Apply histogram equalization
    gray_equ = cv2.equalizeHist(gray_correct)

    # Local adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray_equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
    thresh = cv2.bitwise_not(thresh)

    # Dilation and erosion to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    img_erode = cv2.medianBlur(img_erode, 11)

    # Find contours
    contours, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    filtered_contours = filter_contours(contours)

    # Visualize contours on the original image
    output_image = original.copy()  # Create a copy of the original image
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)  # Draw filtered contours in green

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Contours')
    plt.show()  # Display the image with contours

    # Prepare to count objects per type
    object_counts = {}

    # Iterate through each filtered contour
    for contour in filtered_contours:
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Crop the object from the original image
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = original[y:y+h, x:x+w]

        # Classify the type of object
        object_type = classify_object(cropped_image)

        if object_type:
            normalized_type = object_type.strip().lower()
            object_counts[normalized_type] = object_counts.get(normalized_type, 0) + 1

    # Log the counts per object type
    for obj_type, count in object_counts.items():
        print(f"******Detected {count} {obj_type}(s)")

    return object_counts

def classify_fruit(cropped_image):
    """Classify the object as fruit using the fruit model."""
    app.logger.debug(f"Starting fruit classification for image of shape: {cropped_image.shape}")
    cropped_image = preprocess(cropped_image)  # Preprocess the image
    cropped_image = cropped_image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = fruit_model(cropped_image)

    # Log the predictions
    app.logger.debug(f"Fruit model predictions: {predictions}")  # Log the predictions array    

    return predictions

def classify_object(cropped_image):
    """Classify the object using both models."""
    app.logger.debug(f"Starting classification for image of shape: {cropped_image.shape}")

    # Resize the cropped image to the required shape for the model
    cropped_image_resized = cv2.resize(cropped_image, (224, 224))  # Resize to 224x224

    # Try classifying as fruit first
    fruit_predictions = classify_fruit(cropped_image_resized)
    
    # Log the fruit predictions
    app.logger.debug(f"Fruit predictions: {fruit_predictions}")

    # Check if the prediction confidence is high enough (example threshold)
    if torch.max(fruit_predictions) > 0.5:  # Adjust threshold as needed
        predicted_label_index = torch.argmax(fruit_predictions).item()
        app.logger.debug(f"Classified as fruit: {predicted_label_index}")
        return labels[predicted_label_index]  # Assuming labels are correctly indexed
    
    # If not fruit, use the general model
    pil_image = Image.fromarray(cropped_image_resized)
    input_array = preprocess(pil_image).unsqueeze(0)

    with torch.no_grad():
        predictions = general_model(input_array)
    
    # Get the predicted class label
    predicted_label_index = torch.argmax(predictions).item()
    app.logger.debug(f"Classified object as: {predicted_label_index}")
    return labels[predicted_label_index].strip().lower()

def recognize_barcode(image_path):
    """Recognize barcodes in the given image and return product information."""
    app.logger.info(f"Checking image_path: {image_path}")

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        app.logger.error("Error: Could not read the image.")
        return [], None

    # Decode the barcode(s) in the image
    barcodes = decode(image)
    results = []

    # Process detected barcodes
    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        app.logger.info(f"Decoded barcode: {barcode_data}")
        product_info = lookup_product_by_barcode(barcode_data)

        if product_info:
            results.append({
                'barcode': barcode_data,
                'name': product_info['name'],
                'brand': product_info['brand'],
                'category': product_info['category'],
                'price': product_info['price'],
                'quantity': 0,  # Placeholder for quantity
            })
        else:
            app.logger.warning(f"Product not found for barcode: {barcode_data}")
            results.append({
                'barcode': barcode_data,
                'name': 'Not found',
                'brand': 'N/A',
                'category': 'N/A',
                'price': 'N/A',
                'quantity': 0,  # Placeholder for quantity
            })

    return results, image  # Return results along with the original image for further processing

def recognize_item_without_barcode(image_path, object_counts):
    """Use image recognition to identify an item without a barcode."""
    app.logger.info("Starting item recognition without barcode.")
    
    try:
        img = Image.open(image_path).convert("RGB")  # Load and convert image
        input_array = preprocess(img).unsqueeze(0)  # Apply preprocessing and create a mini-batch

        # Make prediction
        with torch.no_grad():
            predictions = general_model(input_array)
        
        # Get the predicted class label
        predicted_label_index = torch.argmax(predictions).item()
        predicted_label = labels[predicted_label_index]

        app.logger.info(f"Predicted label: {predicted_label}")  # Log the predicted label

        # Get top 3 predictions
        decoded_prediction2s = [(labels[i], predictions[0][i].item()) for i in torch.topk(predictions, 3).indices[0]]

        # Log the predictions for debugging
        app.logger.debug(f"Top 3 Predictions: {decoded_prediction2s}")
        
        # Normalize the predicted label
        normalized_name = predicted_label.strip().lower()
        quantity = object_counts.get(normalized_name, 0)  # Use the count from object_counts

        # Lookup product information by name
        category_info = lookup_product_by_name(predicted_label)

        result = {
            'name': predicted_label,
            'brand': category_info['brand'] if category_info else 'N/A',
            'category': category_info['category'] if category_info else 'N/A',
            'price': category_info['price'] if category_info else 'N/A',
            'quantity': quantity
        }
        
        return [result]  # Return a list with a dictionary
    except Exception as e:
        app.logger.error(f"Error during item recognition: {e}")
        return [{'name': 'Error during recognition', 'brand': 'N/A', 'category': 'N/A', 'price': 'N/A', 'quantity': 0}]

def process_image(image_path):
    """Process the selected image for barcode recognition and object counting."""
    image = cv2.imread(image_path)
    
    if image is None:
        app.logger.error("Error: Could not read the image.")
        return []

    results, _ = recognize_barcode(image_path)

    if not results or all(result['name'] == 'Not found' for result in results):
        app.logger.info("No valid barcodes found, proceeding with image recognition.")
        
        # Count objects in the image
        object_counts = count_objects(image)
        app.logger.info(f"Object counts after image recognition: {object_counts}")

        # Recognize items based on the image content
        results = recognize_item_without_barcode(image_path, object_counts)
    else:
        # If barcodes are found, count objects again for verification
        object_counts = count_objects(image)
        app.logger.info(f"Object counts: {object_counts}")

        # Update results with quantities based on recognized barcodes
        for result in results:
            normalized_name = result['name'].strip().lower()
            quantity = object_counts.get(normalized_name, 0)
            result['quantity'] = quantity

    # Prepare output results
    output_results = prepare_output(results, object_counts)
    app.logger.info(f"Processed results: {output_results}")
    return output_results

def prepare_output(results, object_counts):
    """Prepare the output by filtering valid results and including object counts."""
    output_results = []

    for index, result in enumerate(results, start=1):
        if result.get('name') and result.get('quantity') is not None:
            output_results.append({
                'index': index,
                'barcode': result.get('barcode', 'nil'),
                'name': result.get('name', 'N/A'),
                'brand': result.get('brand', 'N/A'),
                'category': result.get('category', 'N/A'),
                'price': f"${result.get('price', 'N/A')}",
                'quantity': result.get('quantity', 0),
            })
            app.logger.debug(f"Added result to output: {output_results[-1]}")

    if object_counts:
        output_results.append({'object_counts': object_counts})
        app.logger.debug(f"Added object counts to output: {object_counts}")

    app.logger.info(f"Final prepared output: {output_results}")
    return output_results

# Example usage
if __name__ == "__main__":
    image_path = 'G:\\My Drive\\IERG4998\\captured_image.jpg'  # Replace with the actual image path
    results = process_image(image_path)
    print(results)  # Print results for testing