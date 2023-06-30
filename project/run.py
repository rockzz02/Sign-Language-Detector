import cv2
import mediapipe as mp
import pickle
import numpy as np


cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

dictionary = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L'}

while True:
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        for i in range(len(hand_landmarks.landmark)):
            # print(hand_landmarks.landmark[i])
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)
            y_.append(y)
        
        x1 = int(min(x_) * W) - 15
        y1 = int(min(y_) * H) - 15
        x2 = int(max(x_) * W) - 15
        y2 = int(max(y_) * H) - 15
        
        prediction = model.predict([np.asarray(data_aux)])
        predicted_char = dictionary[int(prediction[0])]

        # print(predicted_char)

        cv2.rectangle(frame, (x1, y1), (x2 + 20, y2 + 20), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()