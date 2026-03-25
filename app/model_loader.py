# app/model_loader.py
import os
import tensorflow as tf

class KidneyModel:
    def __init__(self):
        self.img_size = (224, 224)
        # Go up one level from /app to find weights in root
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.weights_path = os.path.join(self.root_dir, "tumor_weights.weights.h5")
        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224,224,3)),
            tf.keras.layers.Rescaling(1./127.5, offset=-1),
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _load_weights(self):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print(f"✅ SUCCESS: Loaded weights from {self.weights_path}")
        else:
            print(f"❌ ERROR: Weights not found at {self.weights_path}")

    def predict(self, data):
        return self.model.predict(data)

kidney_ai = KidneyModel()