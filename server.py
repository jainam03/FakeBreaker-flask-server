from flask import Flask, request, jsonify
import io
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
import soundfile as sf
import numpy as np
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
CORS(
    app,
    origins=["https://fakebreaker-vite-client-dxk5.onrender.com", "http://localhost:5173"],
    supports_credentials=True,
)  # Enable cross-origin requests
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())


# =========================
# Improved Audio Classifier Model
# =========================
class ImprovedAudioClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout_prob=0.5):
        super(ImprovedAudioClassifier, self).__init__()
        # Input: Combined features [1, 148, T_fixed]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # First residual block
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.downsample2 = nn.Conv2d(32, 64, kernel_size=1)  # For residual connection
        self.pool2 = nn.MaxPool2d(2)

        # Second residual block
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.downsample3 = nn.Conv2d(64, 128, kernel_size=1)  # For residual connection
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        # Attention mechanism
        self.attention = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1), nn.Sigmoid())

        # Multiple pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Final classification layers
        self.fc1 = nn.Linear(256 * 2, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # First residual block
        identity2 = self.downsample2(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity2)  # Residual connection
        x = self.pool2(x)
        x = self.dropout(x)

        # Second residual block
        identity3 = self.downsample3(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = F.relu(x + identity3)  # Residual connection
        x = self.pool3(x)
        x = self.dropout(x)

        # Final convolution and pooling
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Apply attention
        att = self.attention(x)
        x = x * att

        # Global pooling strategies
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =========================
# Preprocessing Function
# =========================
def preprocess_audio_file(
    audio_data, sr, target_sample_rate=10000, max_duration=20.0, T_fixed=300
):
    try:
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        waveform = torch.FloatTensor(audio_data).unsqueeze(0)  # shape: [1, L]

        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        # Trim or pad waveform to max_samples
        max_samples = int(target_sample_rate * max_duration)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[1] < max_samples:
            padding = torch.zeros(1, max_samples - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)

        # Compute Mel spectrogram with updated parameters
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)  # log transformation

        # Compute MFCC features with updated parameters
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=target_sample_rate,
            n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        )
        mfcc = mfcc_transform(waveform)

        # Normalize features
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

        # Remove extra channel dimension from both (if present)
        mel_spec = mel_spec.squeeze(0)  # now shape: [128, T]
        mfcc = mfcc.squeeze(0)  # now shape: [20, T]

        # Concatenate along the feature dimension
        combined = torch.cat([mel_spec, mfcc], dim=0)  # shape: [148, T]

        # Force the time dimension to be exactly T_fixed using interpolation
        combined = combined.unsqueeze(0)  # shape: [1, 148, T]
        combined = F.interpolate(
            combined, size=T_fixed, mode="linear", align_corners=False
        )
        combined = combined.squeeze(0)  # shape: [148, T_fixed]

        # Add channel dimension for CNN input -> [1, 148, T_fixed]
        combined = combined.unsqueeze(0)
        return combined

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


# =========================
# Classification Function
# =========================
def classify_audio_clip(file):
    try:
        # Read the file using soundfile
        # audio_data, sr = sf.read(file)
        #Load only first 20 seconds of audio to save memory
        info = sf.info(file)
        sr = info.samplerate
        max_samples = int( sr * 20 )
        audio_data, _ = sf.read(file, stop=max_samples)

        # Preprocess the audio to obtain combined features
        features = preprocess_audio_file(audio_data, sr)
        if features is None:
            return None

        # Add batch dimension: final shape [1, 1, 148, T_fixed]
        features = features.unsqueeze(0)

        # Load the improved model
        model_path = os.environ.get("MODEL_PATH", "./audio_classifier_improved.pth")
        classifier = ImprovedAudioClassifier(n_classes=2)
        classifier.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        classifier.eval()

        # Get predictions
        with torch.no_grad():
            outputs = classifier(features)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        # Convert probabilities to Python float percentages
        fake_prob = float(probs[0] * 100)
        real_prob = float(probs[1] * 100)
        final_label = (
            "FAKE (deepfake)" if fake_prob > real_prob else "REAL (human voice)"
        )

        return fake_prob, real_prob, final_label

    except Exception as e:
        print(f"Error in classification: {e}")
        return None


# =========================
# API Endpoints
# =========================
@app.route("/api/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload():
    print("🔹 Upload route triggered")
    if "file" not in request.files:
        print("⚠️ No file part in request")
        return jsonify(error="No file part in request"), 400

    file = request.files["file"]
    if file.filename == "":
        print("⚠️ No selected file")
        return jsonify(error="No file selected"), 400

    print(f"✅ Received file: {file.filename}")

    # Save file temporarily for processing
    temp_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(temp_path)

    result = classify_audio_clip(temp_path)

    # Remove the temporary file
    os.remove(temp_path)

    if result is not None:
        fake_prob, real_prob, final_label = result
        print(f"✅ Classification result: {final_label} (Fake: {fake_prob:.2f}%)")
        return jsonify(
            {
                "label": final_label,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "note": "Only the first 20 seconds of audio were used for analysis"
            }
        )
    else:
        print("❌ Audio processing error")
        return jsonify(error="Audio processing error"), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify(error="File is too large. Please upload a file under 16 MB.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
