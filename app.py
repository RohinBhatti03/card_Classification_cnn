from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from tensorflow.keras.models import load_model


model = load_model("model.keras")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

app = FastAPI()

def preprocess_image(photo_path):
    img = image.load_img(photo_path, target_size=(128,128), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][idx]
    return index_to_class[idx], float(confidence)

@app.post("/predict")
async def prediction_function(main_photo: UploadFile = File(...)):
    # Save uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(await main_photo.read())
    
    predicted_card, confidence = preprocess_image("temp.jpg")
    return JSONResponse(content={"predicted_card": predicted_card, "confidence": confidence})