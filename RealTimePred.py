import cv2
import numpy as np
import joblib
model = joblib.load("hair_model.pkl")        
scaler = joblib.load("scaler.pkl")            
encoder = joblib.load("label_encoder.pkl")    
from imageProcessing import resize
from skimage.feature import hog


cam = cv2.VideoCapture(0)


if not cam.isOpened():
    print("can't open camera")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("not able to grab frame")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    x1, y1 = w//4, h//4
    x2, y2 = w*3//4, h*3//4
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    imageResize = resize(roi, (64, 64), anti_aliasing=True)
    imageResize = np.clip(imageResize, 0, 1)
    imageResize = imageResize.astype(np.float32)

    features = imageResize.flatten()


    features_scaled = scaler.transform([features])

    pred = model.predict(features_scaled)
    hair_type = encoder.inverse_transform(pred)[0]

    cv2.putText(frame, f"Hair: {hair_type}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Hair Type Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()