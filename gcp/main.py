from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = [
    'Background_without_leaves',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy'
]

BUCKET_NAME = "apurva_tf_models" # Here you need to put the name of your GCP bucket

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model_8_Classes_2",
            "/tmp/model_8_Classes_2",
        )
        model = tf.keras.models.load_model("/tmp/model_8_Classes_2")
        print("Input shape of model:", model.input_shape)
    image = request.files["file"]
    print(f"Received image: {image.filename}")
    img = Image.open(image).convert("RGB")
    img = img.resize((256,256))
    img_batch = np.asarray(img)
    img_batch = np.expand_dims(img_batch, 0)
    print("Shape of input array:", img_batch.shape)

    # image = np.array(
    #     Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    # )

    # image = image/255 # normalize the image in 0 to 1 range

    # img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_batch)

    print("Predictions:",predictions)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * (np.max(predictions[0])), 2)

    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
    
    return {"class": predicted_class, "confidence": confidence}
