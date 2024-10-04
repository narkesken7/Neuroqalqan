import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset
train_dir = '/Users/keshubai/Desktop/CS/Python/Neuroqalqan/Brain_Data_1/Training'
test_dir = '/Users/keshubai/Desktop/CS/Python/Neuroqalqan/Brain_Data_1/Testing'

# Data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating data generators
tr_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

valid_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

# Model definition
base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  #4 classes: pituitary, notumor, glioma, and meningioma
])

# Compiling the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Setting up checkpointing to save the model during training
checkpoint = ModelCheckpoint(
    'model_checkpoint.h5',  # Filepath to save the model
    monitor='val_loss',     # Monitoring validation loss
    save_best_only=True,    # Save the model with the best validation loss
    mode='min',             # Minimize the loss
    verbose=1
)

# adding early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, # Number of epochs with no improvement to wait before stopping
    verbose=1,
    restore_best_weights=True
)

# Training the model with the checkpoint callback
history = model.fit(
    tr_gen,
    epochs=10,
    validation_data=valid_gen,
    shuffle=False,
    callbacks=[checkpoint, early_stopping]
)

# Saving the final model
model.save('final_model.h5')

