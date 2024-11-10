# import os
# import cv2

# # Directory to save data
# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# # Number of classes (26 alphabets + 1 space)
# number_of_classes = 27
# dataset_size = 100

# # Try different camera indices if the default one doesn't work
# cap = cv2.VideoCapture(0)  # Change the index if necessary

# if not cap.isOpened():
#     print("Error: Unable to open camera")
#     exit()

# # Loop through the classes (0-25 for A-Z, 26 for space)
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))

#     label = chr(65 + j) if j < 26 else 'SPACE'  # A-Z for 0-25, SPACE for 26
#     print(f'Preparing to collect data for class {label}...')

#     # Indicate the camera is ready to start collecting images
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame. Exiting...")
#             break

#         cv2.putText(frame, 'Press "Q" to start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)
#         cv2.imshow('frame', frame)

#         if cv2.waitKey(25) == ord('q'):
#             print(f'Starting to capture images for class {label}...')
#             break

#     # Start capturing images
#     counter = 0
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame. Exiting...")
#             break

#         cv2.putText(frame, f'Capturing image {counter+1}/{dataset_size}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.imshow('frame', frame)

#         # Save the captured frame to the dataset
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
#         counter += 1

#         if cv2.waitKey(25) == ord('q'):  # Allow early exit with 'q'
#             print('Capture stopped by user.')
#             break

#     print(f'Finished collecting data for class {label}.')

# cap.release()
# cv2.destroyAllWindows()
# print("All data collection complete.")

#-------------------------------------------------------------------------------------------------------------------------------------------------

import os
import cv2

# Directory to save data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (26 alphabets + 1 space)
number_of_classes = 27
dataset_size = 100

# Try different camera indices if the default one doesn't work
cap = cv2.VideoCapture(0)  # Change the index if necessary

if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

# Loop through the classes (0-25 for A-Z, 26 for space)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    label = chr(65 + j) if j < 26 else 'SPACE'  # A-Z for 0-25, SPACE for 26
    print(f'Preparing to collect data for class: {label}...')

    # Indicate the camera is ready to start collecting images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.putText(frame, 'Press "Q" to start capturing images', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f'Current Class: {label}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            print(f'Starting to capture images for class: {label}...')
            break

    # Start capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.putText(frame, f'Capturing image {counter + 1}/{dataset_size}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save the captured frame to the dataset
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1

        if cv2.waitKey(25) == ord('q'):  # Allow early exit with 'q'
            print('Capture stopped by user.')
            break

    print(f'Finished collecting data for class: {label}.')

cap.release()
cv2.destroyAllWindows()
print("All data collection complete.")
