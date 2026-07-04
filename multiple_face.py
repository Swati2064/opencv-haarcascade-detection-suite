import cv2

# Load the Haar cascade files for face and eye detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Check if the cascade files were loaded properly
if face_classifier.empty():
    print("Error: Could not load face cascade classifier.")
    exit()

if eye_classifier.empty():
    print("Error: Could not load eye cascade classifier.")
    exit()


# Function to detect faces and eyes
def detect_faces_and_eyes(gray, frame):

    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
            )

    return frame


# Open webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam started successfully... Press 'q' to quit.")

while True:

    ret, frame = video_capture.read()

    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = detect_faces_and_eyes(gray, frame)

    cv2.imshow("Face and Eye Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()