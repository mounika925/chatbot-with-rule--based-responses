import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
import pickle
import os

# --------- STEP 1: Feature Extraction from Image ---------
def extract_features(img_path):
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.get_layer('avg_pool').output)

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = model_new.predict(x)
    return feature

# --------- STEP 2: Generate Caption ---------
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# --------- STEP 3: Load Model & Tokenizer ---------
# Update these paths to your trained model and tokenizer
model_path = 'caption_model.h5'         # Trained model
tokenizer_path = 'tokenizer.pkl'        # Saved tokenizer

model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34  # Adjust based on your training

# --------- STEP 4: Run on an Image ---------
img_path = 'example.jpg'  # Replace with your image path
photo = extract_features(img_path)
caption = generate_desc(model, tokenizer, photo, max_length)

print("Generated Caption:", caption)
