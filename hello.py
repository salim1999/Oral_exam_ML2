from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    # Set the target size
    target_size = (192, 192, 3)

    model = load_model('model/my_model20230609172115.h5')

  

    if 'photo' in request.files:
        photo = request.files['photo']
        image = Image.open(photo)
        resized_image = image.resize(target_size[:2])
        resized_image = resized_image.convert('RGB')

        img_array = tf.keras.utils.img_to_array(resized_image)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict_on_batch(img_array).flatten()

        prediction = tf.where(prediction < 0.5, 0, 1)
        class_names = ['cat','dog']

        return  "This image most likely belongs to " + class_names[prediction[0]]