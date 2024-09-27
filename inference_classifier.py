# import pickle

# import cv2
# import mediapipe as mp
# import numpy as np

# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {0: 'A', 1: 'B', 2: 'C'}
# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         predicted_character = labels_dict[int(prediction[0])]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     cv2.imshow('frame', frame)
#     cv2.waitKey(1)


# cap.release()
# cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------------------------


# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyttsx3  # Import the text-to-speech library

# # Load the trained model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize video capture
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Update this dictionary with all classes (0-26 for A-Z and SPACE)
# labels_dict = {i: chr(65 + i) for i in range(26)}
# labels_dict[26] = 'SPACE'

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame. Exiting...")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#             # Calculate bounding box
#             x1 = int(min(x_) * W) - 10
#             y1 = int(min(y_) * H) - 10
#             x2 = int(max(x_) * W) - 10
#             y2 = int(max(y_) * H) - 10

#             # Make prediction
#             prediction = model.predict([np.asarray(data_aux)])

#             predicted_character = labels_dict[int(prediction[0])]

#             # Draw bounding box and text
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                         cv2.LINE_AA)

#             # Text-to-speech feedback
#             engine.say(predicted_character)
#             engine.runAndWait()

#     cv2.imshow('frame', frame)

#     # Exit condition
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#------------------------------------------

import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # Import the text-to-speech library

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Update this dictionary with all classes (0-26 for A-Z and SPACE)
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = 'SPACE'

# Initialize a list to hold the recognized characters
recognized_sentence = []

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

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
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            # If the predicted character is SPACE, add a space to the sentence
            if predicted_character == 'SPACE':
                recognized_sentence.append(' ')
            else:
                recognized_sentence.append(predicted_character)

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            # Text-to-speech feedback
            engine.say(predicted_character)
            engine.runAndWait()

    # Display the constructed sentence
    sentence = ''.join(recognized_sentence)
    cv2.putText(frame, sentence, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
