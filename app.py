from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
tomato_model = tf.keras.models.load_model('models/tomato_disease_model.h5')
mango_model = tf.keras.models.load_model('models/mango_disease_model.h5')
apple_model = tf.keras.models.load_model('models/apple_disease_model.h5')

# Define class names
tomato_class_names = ['tomato_reject', 'tomato_ripe', 'tomato_unripe']
mango_class_names = ['mango_Alternaria', 'mango_anthracnose', 'mango_BlackMouldRot', 'mango_healthy', 'mango_stem&rot']
apple_class_names = ['apple_BOTCH', 'apple_NORMAL', 'apple_ROT', 'apple_SCAB']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/upload/<fruit>', methods=['GET', 'POST'])
def upload(fruit):
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded image
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Make prediction
            predicted_class, predicted_probability = make_prediction(fruit, file_path)
            
            # Pass the result to the result page
            return render_template('result.html', fruit=fruit, filename=filename, predicted_class=predicted_class, predicted_probability=predicted_probability)
    return render_template('upload.html', fruit=fruit)

def make_prediction(fruit, image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Select the appropriate model and class names
    if fruit.lower() == 'tomato':
        model = tomato_model
        class_names = tomato_class_names
    elif fruit.lower() == 'mango':
        model = mango_model
        class_names = mango_class_names
    elif fruit.lower() == 'apple':
        model = apple_model
        class_names = apple_class_names
    else:
        return None, None
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    
    return predicted_class, predicted_probability * 100

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
