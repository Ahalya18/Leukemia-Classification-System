import os
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def generate_model(save_path="leukonet.keras"):
    # Ensure current working directory is correct when saving
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Load base model
    print("Loading MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add classification head for ALL and AML
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.save(save_path)
    print(f"Dummy model generated successfully and saved to {save_path}")

if __name__ == "__main__":
    # We save to the current directory relative to the script execution or an absolute path
    # Let's just save to model/leukonet.keras
    generate_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leukonet.keras"))
