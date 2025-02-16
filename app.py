from flask import Flask, render_template, request, flash, redirect, url_for
import sys
import os
import logging

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from barcode_functions
from barcode_functions import capture_image, recognize_barcode, recognize_item, count_quantity_from_image
from image_processing import process_image  # Import the process_image function

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management (flashing messages)

# Ensure that the uploads directory exists
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    try:
        result = capture_image()  # Call the image capture function
        return render_template('capture.html', result=result)  # Pass result to the template
    except Exception as e:
        logging.error(f"Error in capture: {str(e)}")
        flash('Error capturing image. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        item_image = request.files.get('file')
        if item_image and allowed_file(item_image.filename):
            uploads_directory = 'uploads'
            os.makedirs(uploads_directory, exist_ok=True)
            #try:
            image_path = os.path.join(uploads_directory, item_image.filename)
            item_image.save(image_path)  # Save the uploaded file temporarily
            #result = "Image processed successfully!"
            result = recognize_item(image_path)  # Call the recognize_item function
    
            # Clean up the uploaded file
            #os.remove(image_path)
            # Log the recognition results for debugging
            logging.debug(f"Recognization Results: {result}")

            return render_template('recognize.html', result=result)  # Render results
            #except Exception as e:
                #logging.error(f"Error in recognize: {str(e)}")
                #flash('Error recognizing item. Please try again.', 'error')
                #return redirect(url_for('recognize'))
        else:
            flash('No file selected or Invalid file type. Please upload an image.', 'error')
            return redirect(url_for('recognize'))
    return render_template('recognize_form.html')  # Template for the form

@app.route('/count', methods=['GET', 'POST'])
def count():
    if request.method == 'POST':
        item_image = request.files.get('item_image')
        logging.info(f"File received: {item_image.filename}")
        if item_image and allowed_file(item_image.filename):
            try:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], item_image.filename)
                item_image.save(image_path)  # Save the uploaded file temporarily
                logging.info(f"File saved to: {image_path}")

                results = process_image(image_path)  # Use process_image for counting and recognition
                logging.debug(f"Processed results: {results}")

                return render_template('count.html', result=results)  # Render results
            except Exception as e:
                logging.error(f"Error in count: {str(e)}")
                flash('Error processing the image. Please try again.', 'error')
                return redirect(url_for('count'))
        else:
            flash('Invalid file type. Please upload an image.', 'error')
    return render_template('count_form.html')  # Template for the form

@app.route('/leave')
def leave():
    return render_template('leave.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash("No file part", 'error')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash("No selected file or invalid file type", 'error')
        return redirect(url_for('index'))

    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)  # Save the file to the upload folder

        # Recognize barcodes in the uploaded image
        barcodes = recognize_barcode(image_path)

        # Clean up the uploaded file
        ##os.remove(image_path)

        # Return the results
        if barcodes:
            return f"Detected Barcodes: {', '.join(barcodes)}"
        else:
            return "No barcodes found in the uploaded image."
    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        flash('Error processing the image. Please try again.', 'error')
        return redirect(url_for('index'))

