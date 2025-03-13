import cv2

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.reshape(1, 48, 48, 1) / 255.0
    return image
