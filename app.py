import os
import sys

# --- SILENCE TENSORFLOW LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIG ---
SERVICE_MAP = {
    "Cleaner": ["clean","dust","mop","wash","sweep","garbage","trash","messy","housekeeping","cleaner"],
    "Electrician": ["fan","light","bulb","switch","socket","wire","fuse","mcb","shock","current","power","voltage","electrician"],
    "Painter": ["paint","wall","color","whitewash","stain","brush","roller","exterior","interior","painter"],
    "Salon": ["hair","cut","shave","beard","facial","massage","makeup","beauty","style","grooming","salon"],
    "Carpenter": ["wood","door","window","furniture","table","chair","bed","lock","handle","cupboard","shelf","carpenter"],
    "Mechanic": ["car","bike","scooter","vehicle","engine","brake","clutch","gear","oil","tire","puncture","start","mechanic"]
}

# Load model once
try:
    model = tf.keras.models.load_model('profix_brain_model')
except:
    model = None


# --- AI FUNCTION ---
def detect_service(input_text):
    text = input_text.lower()

    # Keyword matching
    for service, words in SERVICE_MAP.items():
        for w in words:
            if w in text:
                return service

    # AI fallback
    if model:
        try:
            pred = model.predict([text], verbose=0)[0]

            if np.max(pred) < 0.6:
                return "I am trained for Profix AI services only"

            return "Mechanic" if pred[0] > pred[1] else "Electrician"

        except:
            return "Model error"

    return "I am trained for Profix AI services only"


# --- ROUTES ---
@app.route("/")
def home():
    return "Profix AI Chatbot Running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message","")

    reply = detect_service(msg)

    return jsonify({"reply": reply})


# Required for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
