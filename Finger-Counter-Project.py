import cv2
import mediapipe as mp
import os
import time

# ---------------- Hand Detection Setup ----------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ---------------- Load Overlay Images with Safe Mapping ----------------
folder_path = r"C:\Users\ABBAS COMPUTERS\Desktop\FingerImages"
overlay_dict = {}

for img_name in os.listdir(folder_path):
    if img_name.lower().endswith(".png"):
        try:
            count = int(img_name.split('.')[0])  # e.g., 2.png -> 2
            overlay_dict[count] = cv2.imread(os.path.join(folder_path, img_name))
        except:
            pass

print(f"Loaded overlay images for counts: {list(overlay_dict.keys())}")

# ---------------- Webcam Setup ----------------
cap = cv2.VideoCapture(0)
tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

pTime = 0
finger_threshold = 20       # y-threshold for index to pinky
thumb_threshold = 20        # x-threshold for thumb tip vs IP joint

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if lmList:
        fingers = []

        # ----- Thumb Detection (tip vs IP joint) -----
        thumb_tip_x, thumb_tip_y = lmList[4][1], lmList[4][2]
        thumb_ip_x, thumb_ip_y = lmList[3][1], lmList[3][2]
        fingers.append(1 if thumb_tip_x - thumb_ip_x > thumb_threshold else 0)

        # ----- Index-Pinky Fingers -----
        for id in range(1, 5):
            tip_y = lmList[tipIds[id]][2]
            pip_y = lmList[tipIds[id]-2][2]
            fingers.append(1 if tip_y < pip_y - finger_threshold else 0)

        totalFingers = fingers.count(1)

        # ----- Overlay Image -----
        if totalFingers in overlay_dict:
            overlay = overlay_dict[totalFingers]
            frame_h, frame_w, _ = img.shape
            overlay = cv2.resize(overlay, (frame_w // 2, frame_h // 2))
            h, w, c = overlay.shape
            img[0:h, 0:w] = overlay

        # ----- Show Finger Count -----
        if totalFingers > 0:
            cv2.putText(img, str(totalFingers), (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    # ----- FPS -----
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
