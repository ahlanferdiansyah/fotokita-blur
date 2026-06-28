import cv2
import mediapipe as mp

#detect tangan
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def finger_up(tip, pip, landmarks):
    return landmarks[tip].y < landmarks[pip].y


def is_peace(landmarks):

    index_up = finger_up(8, 6, landmarks)
    middle_up = finger_up(12, 10, landmarks)

    ring_up = finger_up(16, 14, landmarks)
    pinky_up = finger_up(20, 18, landmarks)

    return (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    )

#open camera
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    hand_result = hands.process(rgb)

    peace_detected = False

    if hand_result.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils

        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Menggambar kerangka/titik tangan dengan warna kuning (BGR: 0, 255, 255)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
            )

            if is_peace(hand_landmarks.landmark):
                peace_detected = True
                # Kita tidak break di sini agar semua tangan yang terdeteksi tetap digambar


    #blur efek

    if peace_detected:

        frame = cv2.GaussianBlur(
            frame,
            (61, 61),
            0
        )

    cv2.imshow(
        "Peace Blur",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
