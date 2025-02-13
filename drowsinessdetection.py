#headerSection
import cv2
import os
import numpy as np
import dlib
from imutils import face_utils
import random
#-----------------------------------------------------------------------------
# A random alert message is selected from lst and stored in label2.
lst=["Be Alert!.A Man Sleep in his Driving.","Back to Drive. Be Alert!","Can't Sleep-Think about your future.","Be Alert!,Don't Sleep at All"]
label2=random.choices(lst)
#webcamSection
cap=cv2.VideoCapture(0)   # Capture video from default Camera ,while cv.VideoCapture(1) is used when to capture video from external camera. 

# Set Video frame size to 640*480
cap.set(3,640)            # Set Video Width = 640 pixels
cap.set(4,480)            # Set Video Height = 480 pixels 

#Initializing the face detector and landmark detector
#  detector: Detects faces using Dlib's pre-trained frontal face detector.
#  predictor: Loads the 68-point facial landmark model (shape_predictor_68_face_landmarks.dat)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#status marking for current state
sleep = 0
drowsy = 0
active = 0
color=(0,0,0)
# Eye Blink Calculation Function
def compute(ptA,ptB):
        dist = np.linalg.norm(ptA - ptB)  # Euclidean distance between two points
        return dist

def blinked(a,b,c,d,e,f):
   up = compute(b,d) + compute(c,e)    # Sum of vertical distances
   down = compute(a,f)                 # Horizontal distance
   ratio = up/(2.0*down)               # Eye Aspect Ratio (EAR)

    #Checking if it is blinked
   if(ratio>0.25):                     # Eye Open (Active)
             return 2
   elif(ratio>0.21 and ratio<=0.25):   # Drowsy
            return 1
   else:                               # Eye Closed (Sleep)
           return 0

# Loads UI elements (images) from the Resources/bg folder.
imgback=cv2.imread("background.png")
#to reterive data of model from folder
modefolder="Resources/Resources/bg"
listmodefolder=os.listdir(modefolder)
imgmodelist=[]
for path in listmodefolder:
    imgmodelist.append(cv2.imread(os.path.join(modefolder,path)))
print(len(imgmodelist))



# Real-time Detection
while True:
    success,img=cap.read()                                # Capture frame from webcam
    imgback[162:162 + 480,55:55 + 640]=img                # Overlay webcam feed on background
    imgback[44:44 + 633,808:808 + 414]=imgmodelist[0]     # Displays the default UI image.

    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(imgback, cv2.COLOR_BGR2GRAY)    # Convert to grayscale - improves detection speed
    faces = detector(gray)                              # Detects faces using dlib.
    #detected face in faces array
    for face in faces:
        x1 = face.left()                                # Extracts facial coordinates (x1, y1, x2, y2).
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = imgback.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)              # Detects facial landmarks using the predictor
        landmarks = face_utils.shape_to_np(landmarks)  # # Convert to NumPy array

        # The numbers are actually the landmarks which will show eye
        # Calls blinked() function to determine if both eyes are open, drowsy, or closed.
        left_blink = blinked(landmarks[36],landmarks[37], 
           landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
          landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
               sleep+=1
               drowsy=0
               active=0
               if(sleep>6):          # If eyes closed for more than 6 frames → Sleep detected.
                   #414x640imagesize
                      imgback[162:162 + 480,55:55 + 640]=img                # Sleep warning
                      imgback[44:44 + 635,808:808 + 414]=imgmodelist[1]
                      color = (0,0,255)                                     # Red Color

        elif(left_blink==1 or right_blink==1):
                sleep=0
                active=0
                drowsy+=1
                if(drowsy>6):        # If drowsy state detected for 6 frames → Drowsiness detected.

                       
                       imgback[162:162 + 480,55:55 + 640]=img
                       imgback[44:44 + 635,808:808 + 417]=imgmodelist[3]
                       color = (0,0,255)

        else:
               drowsy=0
               sleep=0
               active+=1
               if(active>6):   # → Active detected.
                      
                      imgback[162:162 + 480,55:55 + 640]=img             
                      imgback[44:44 + 633,808:808 + 414]=imgmodelist[2]  
                      color = (0,255,0)                                  
          
        #cv2.putText(imgback, status, (910,550), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color,3)
        label2=str(label2)
        cv2.putText(imgback, label2[2:-2], (850,180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0,105), 2, cv2.LINE_AA)
        
    for n in range(0, 68):
           (x,y) = landmarks[n]
           cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)




    cv2.imshow("DriverAlertSystem",imgback)
    cv2.waitKey(1)
