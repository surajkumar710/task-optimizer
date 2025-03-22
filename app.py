from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import torch
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
import os
import azure.cognitiveservices.speech as speechsdk  # Add this import

app = Flask(__name__)

# Load pre-trained models
# Face Emotion Detection (using a placeholder model)
face_emotion_model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
face_emotion_model.eval()

# Text and Speech Emotion Analysis (using Hugging Face Transformers)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Emotion labels for face detection (example labels)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Text Emotion Analysis
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    data = request.json
    text = data.get("text", "")
    if text:
        try:
            result = sentiment_pipeline(text)
            return jsonify({"emotion": result[0]["label"], "score": result[0]["score"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No text provided"}), 400

# Face Emotion Analysis
@app.route("/analyze_face", methods=["POST"])
def analyze_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    try:
        # Read the image file and decode it
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load the face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # Placeholder for face emotion detection
        emotion = "Happy"  # Replace with actual logic
        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Speech Emotion Analysis (using Azure Speech Service)
@app.route("/analyze_speech", methods=["POST"])
def analyze_speech():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    temp_path = "temp_audio"
    try:
        # Save the uploaded file temporarily
        file.save(temp_path)

        # Use Azure Speech Service
        speech_config = speechsdk.SpeechConfig(subscription="YOUR_AZURE_SUBSCRIPTION_KEY", region="YOUR_AZURE_REGION")
        audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text
            # Analyze the text for emotion
            result = sentiment_pipeline(text)
            return jsonify({"emotion": result[0]["label"], "score": result[0]["score"]})
        else:
            return jsonify({"error": "Speech recognition failed"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)