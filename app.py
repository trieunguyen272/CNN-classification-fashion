import os
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "fashion_mnist_cnn_model.h5")

image_size = 28


def preprocess_image(image):
    image = cv2.bitwise_not(image)
    image = tf.expand_dims(image, -1)
    image = tf.divide(image, 255)
    image = tf.image.resize(image,
                            [image_size, image_size])
    image = tf.reshape(image,
                       [1, image_size, image_size, 1])
    return image


def load_and_preprocess_image(path):
    image = cv2.imread(path, 0)
    return preprocess_image(image)


def classify(model, image_path):

    preprocessed_image = load_and_preprocess_image(image_path)
    labelNames = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    probs = cnn_model.predict(preprocessed_image)
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]
    return label, probs[0].max()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(cnn_model, upload_image_path)

        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
