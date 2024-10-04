import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Loading the trained model
model = load_model('final_model.h5')


# Defining function to load and preprocess an image
def preprocess_image(image_path, target_size=(299, 299)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Scale the pixel values between 0 and 1
    return image


# Predict the class of a specific image and display the image with a professional layout
def predict_and_display_image(model, image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]  # Get the class index with the highest score
    class_labels = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary' ]  # Ensure class names are capitalized
    predicted_class = class_labels[class_idx]
    confidence = np.max(prediction)  # Get the confidence score of the predicted class

    # Display the image with a professional layout
    image = load_img(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')

    # Enhancing visual appeal with a professional title
    plt.title(
        f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}",
        fontsize=16,
        fontweight='bold',
        color='#34495E',  # Professional dark grey color
        loc='center',
        pad=20,
    )

    # Adding a frame for aesthetics
    plt.gca().set_frame_on(True)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)

    plt.show()


# Example usage:
image_path = '/Users/keshubai/Desktop/CS/Python/Neuroqalqan/Brain_Data_1/Testing/notumor/Te-noTr_0007.jpg'
predict_and_display_image(model, image_path)
