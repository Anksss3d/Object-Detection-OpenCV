# Import Required Library
import cv2

# Read all class names from file
classes= []
file = 'coco.names'
with open(file,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Configuration file of the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Weights of the trained model
weightsPath = 'frozen_inference_graph.pb'

# Initialize the network with predefined configuration and weights
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize Video Stream for video capture from web camera
videoStream = cv2.VideoCapture(0)
videoStream.set(3,1280)
videoStream.set(4,720)
videoStream.set(10,70)


# Run object detection on web camera video stream untill excape key is pressed
while True:
    # Capture Image
    success,img = videoStream.read()

    # Find objects, its probability and box parameters in the captured image.
    classIds, confs, bbox = net.detect(img,confThreshold=0.6)

    # Generate boxes around obejcts in image
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classes[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # View image with boxes
    cv2.imshow("Output",img)

    # check if pressed key is Escape
    if cv2.waitKey(1) == 27:
        break

# Release video stream and close all windows
videoStream.release()
cv2.destroyAllWindows()