import cv2
import numpy as np
import tensorflow as tf

# Load the model and labels
model = tf.keras.models.load_model("sign_model.keras")

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) - a box in the center
    x1, y1, x2, y2 = 150, 100, 450, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    img = cv2.resize(roi, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    predicted_label = labels[class_index]

    # Draw box and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    text = f"{predicted_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("Hand Sign Detection - Press Q to Quit", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
