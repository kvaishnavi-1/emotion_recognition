from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    "dataset/fer2013/train",
    target_size = (48, 48),
    color_mode = "grayscale",
    batch_size = 64,
    class_mode = "categorical"
)

test_generator = test_datagen.flow_from_directory(
    "dataset/fer2013/test",
    target_size = (48, 48),
    color_mode = "grayscale",
    batch_size = 64,
    class_mode = "categorical"
)

model = Sequential([
    Conv2D(64, (3, 3), activation = 'relu', input_shape = (48,48,1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(7, activation = 'softmax') 
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_generator, epochs = 20, validation_data = test_generator)

model.save("models/emotion_model.h5")
