from ultralytics import YOLO
import cv2


# Load a pre-trained YOLO model
model = YOLO('/Users/dxg45/NerdSuff/PersonalProjects/Gesture-recognition/ObjectRecognition/f1t_car_detection/f1t_car_detection12/weights/best.pt')

# Run inference on an online video stream
# Set stream=True for memory-efficient processing
results = model.predict(source='https://www.youtube.com/watch?v=lbP01VaWrVU', show=True, stream=True)

# Process results
for result in results:
    # Use result.plot() to get an annotated BGR image (OpenCV compatible)
    annotated_frame = result.plot()

    # Display the frame using OpenCV
    cv2.imshow("Ultralytics YOLO Stream", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up resources
cv2.destroyAllWindows()
