import cv2
import numpy as np
import handtrackingmodule as htm
import time
import autopy
import threading

# Configuration
wCam, hCam = 720, 640
frameR = 100  # Frame Reduction
smoothening = 7
click_cooldown = 0.3  # seconds between clicks
display_fps = True
show_finger_markers = True

# Initialize variables
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
last_click_time = 0
running = True

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS if supported by camera
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available

# Set the window to full screen
cv2.namedWindow("AI Virtual Mouse", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("AI Virtual Mouse", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize detector and get screen dimensions
detector = htm.handDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
wScr, hScr = autopy.screen.size()


# Function for mouse operations to be run in separate thread
def move_mouse(x, y):
    try:
        autopy.mouse.move(x, y)
    except Exception as e:
        print(f"Mouse movement error: {e}")


def click_mouse():
    try:
        autopy.mouse.click()
    except Exception as e:
        print(f"Mouse click error: {e}")


def process_frame():
    global pTime, plocX, plocY, clocX, clocY, last_click_time

    # Capture frame
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        return None

    # Process frame with lower resolution for speed
    img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img_small = detector.findHands(img_small)
    lmList, bbox = detector.findPosition(img_small, draw=show_finger_markers)

    # Scale coordinates back to original resolution
    if lmList:
        for i in range(len(lmList)):
            lmList[i][1] *= 2
            lmList[i][2] *= 2

    # Resize image back to original for display
    img = cv2.resize(img_small, (wCam, hCam))

    # Draw control area
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    # Process hand if detected
    if lmList and len(lmList) >= 21:  # Ensure we have all landmarks
        fingers = detector.fingersUp()

        if fingers:  # Check if fingers detection was successful
            # Get index and middle finger positions
            x1, y1 = lmList[8][1:]  # Index finger
            x2, y2 = lmList[12][1:]  # Middle finger

            # Moving mode - Index finger up, middle finger down
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert coordinates with improved mapping
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Apply smoothing with dynamic adjustment
                smoothening_factor = max(1, min(smoothening, 10))  # Constrain between 1-10
                clocX = plocX + (x3 - plocX) / smoothening_factor
                clocY = plocY + (y3 - plocY) / smoothening_factor

                # Move mouse in separate thread to avoid blocking
                mouse_x, mouse_y = wScr - clocX, clocY  # Flip X for intuitive control
                threading.Thread(target=move_mouse, args=(mouse_x, mouse_y), daemon=True).start()

                # Visual feedback
                if show_finger_markers:
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

                # Update previous positions
                plocX, plocY = clocX, clocY

            # Clicking mode - Both index and middle fingers up
            elif fingers[1] == 1 and fingers[2] == 1:
                # Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img, draw=show_finger_markers)

                # Click if distance short and cooldown period passed
                current_time = time.time()
                if length < 40 and (current_time - last_click_time) > click_cooldown:
                    if show_finger_markers:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

                    # Click in separate thread
                    threading.Thread(target=click_mouse, daemon=True).start()
                    last_click_time = current_time

    # Calculate and display FPS
    if display_fps:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    return img


# Main loop
try:
    while running:
        img = process_frame()
        if img is None:
            break

        cv2.imshow("AI Virtual Mouse", img)

        # Check for exit key
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q to quit
            running = False

except KeyboardInterrupt:
    print("Program interrupted by user")
finally:
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")
