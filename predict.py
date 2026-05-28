import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "cat_dog_model.h5"
)

# User inputs
animal = input("Enter animal name: ").lower()
sound = input("Enter animal sound: ").lower()

# Correct combinations
valid_sounds = {
    "cat": "meow",
    "dog": "bow"
}

# Check sound
if animal in valid_sounds and sound == valid_sounds[animal]:

    # Image path
    img_path = input("Enter image path: ")

    # Load image
    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Process image
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print("DOG")
    else:
        print("CAT")

else:
    print("ERROR: Wrong animal sound combination!")