#=======================================================================================================================================================================
#                                                   << Emotions Detection Code >>
#======================================================================================================================================================================


import cv2
from fer import FER
                                                                                    
detector = FER(mtcnn=True)                                                          # Initialize the FER model

cap = cv2.VideoCapture(0)                                                           # Open webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

printed_emotion = False                                                             # Flag to indicate if we've printed the emotion already

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = detector.detect_emotions(frame)                                       # Detect emotions in the frame
    
    if results:
        top_emotion, top_score = detector.top_emotion(frame)
        
        if not printed_emotion:                                                     # Print the emotion only once
            print(f"Top emotion: {top_emotion}")
            printed_emotion = True
        
        for result in results:                                                      # Draw bounding box and emotion on the frame
            (x, y, w, h) = result["box"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, top_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

                        
#==============================================================================================================================================================================
#                                               << EMOTION DETECTION WITH % >> 
#===============================================================================================================================================================================


# import cv2
# from fer import FER

# detector = FER(mtcnn=True)                                                              # Initialize the FER model

# cap = cv2.VideoCapture(0)                                                               # Open webcam

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Detect emotions in the frame
#     results = detector.detect_emotions(frame)
    
#     if results:
#         top_emotion, top_score = detector.top_emotion(frame)
#         print(f"Top emotion: {top_emotion} ({top_score*100:.2f}%)")
        
#         for result in results:                                                          # Draw bounding box and emotion on the frame
#             (x, y, w, h) = result["box"]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, f'{top_emotion}: {top_score*100:.2f}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
#     cv2.imshow('Webcam Feed', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





