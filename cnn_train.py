import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# Dataset Path
# ==========================
dataset_path = "PlantVillage"

# ==========================
# Hyperparameters (FAST)
# ==========================
IMAGE_SIZE = (128, 128)      # 224 -> 128 (Fast)
BATCH_SIZE = 64              # 32 -> 64
EPOCHS = 5                   # 20 -> 5

# ==========================
# Data Generator
# ==========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation Data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================
# Simple CNN (FAST)
# ==========================
model = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape=(128,128,3)),

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save
model.save("model.keras")

print("✅ Training Completed")
print("✅ Model Saved")