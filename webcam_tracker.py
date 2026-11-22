import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
prev_tip_y = None

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        twitch_detected = False
        delta = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                index_tip = hand_landmarks.landmark[8]
                dist = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)

                if prev_tip_y is not None:
                    delta = abs(dist - prev_tip_y)
                    if delta > 0.008: twitch_detected = True
                prev_tip_y = dist

        status = "MICRO-TWITCH VERIFIED" if twitch_detected else "Monitoring..."
        print(f"Status: {status} | Displacement: {delta:.4f}")

cap.release()
