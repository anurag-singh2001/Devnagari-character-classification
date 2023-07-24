import numpy as np 
import pandas as pd 
import os
import base64
from flask import Flask, request, render_template
import cv2
import numpy as np
from keras.models import load_model

train = pd.read_csv("data/data.csv")
train=pd.DataFrame(train)
Column_name=train["character"].to_numpy()

#print(Column_name)

app = Flask(__name__)
model = load_model('model/model.h5')  # Load your trained model

def preprocess_image(image):
    # Preprocess the image as shown in the previous code snippet
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        test_image = preprocess_image(image)

        # Make predictions using the model
        predictions = model.predict(np.expand_dims(test_image, axis=0))
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = Column_name[predicted_class_index]

        # Convert the image to base64 for displaying in the result page
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template('result.html', image_data=img_base64, predicted_class=predicted_class_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)