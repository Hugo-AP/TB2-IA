import cv2.cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, models, layers

# Inicializamos la data de Cifar10
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# Hacemos el escalamiento de la data
training_images, testing_images = training_images / 255, testing_images / 255

# Definimos los labels de las clases
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def training_neuron():
    # Realizamos el recorrido de las primeras 16 imágenes
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[training_labels[i][0]])

    # plt.show()

    # Reducimos la cantidad de imágenes con las que alimentamos la red neuronal
    training_images_train = training_images[:20000]
    training_labels_train = training_labels[:20000]
    testing_images_train = testing_images[:4000]
    testing_labels_train = testing_labels[:4000]

    # Construímos la red neuronal
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Función fit
    model.fit(training_images_train, training_labels_train, epochs=10,
              validation_data=(testing_images_train, testing_labels_train))

    loss, accuracy = model.evaluate(testing_images_train, testing_labels_train)
    print(f"Loss:{loss}")
    print(f"Accuracy:{accuracy}")

    # Guardamos
    model.save('image_classifier.model')


def predict(image):
    # Cargamos el image_classifier.model
    model_func = models.load_model('image_classifier.model')
    # Leemos la imagen de interés
    img = cv.imread('static/test-images/' + image + '.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img, cmap=plt.cm.binary)

    # Resultado de la predicción auto o camión
    prediction = model_func.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    image = class_names[index]
    return image


from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home_page():
    # training_neuron()
    return render_template('home.html')


@app.route('/result-car', methods=['POST', 'GET'])
def result_car():
    output = request.form.to_dict()
    car = output["car"]
    image_name = car + '.jpg'
    car_output = predict(car)
    return render_template('home.html', car=car_output, imageCar=image_name)


@app.route('/result-truck', methods=['POST', 'GET'])
def result_truck():
    output = request.form.to_dict()
    truck = output["truck"]
    truck_output = predict(truck)
    image_name = truck + '.jpg'
    return render_template('home.html', truck=truck_output, imageTruck=image_name)


if __name__ == "__main__":
    app.run(debug=True)
