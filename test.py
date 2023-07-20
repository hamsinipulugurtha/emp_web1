import cv2

# Get the video capture object for the camera
cap = cv2.VideoCapture(0)

# Open the camera
cap.open()

# Check if the camera is opened successfully
if cap.isOpened():
    # Start capturing frames from the camera
    while True:
        # Capture the frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()

# Close all windows
cv2.destroyAllWindows()
