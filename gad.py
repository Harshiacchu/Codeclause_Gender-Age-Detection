import cv2
import math
import argparse


def highlightFace(net, frame, conf_threshold=0.7):
    focv=frame.copy()
    fh=focv.shape[0]
    fw=focv.shape[1]
    blob=cv2.dnn.blobFromImage(focv, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    det=net.forward()
    fb=[]
    for i in range(det.shape[2]):
        confidence=det[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(det[0,0,i,3]*fw)
            y1=int(det[0,0,i,4]*fh)
            x2=int(det[0,0,i,5]*fw)
            y2=int(det[0,0,i,6]*fh)
            fb.append([x1,y1,x2,y2])
            cv2.rectangle(focv, (x1,y1), (x2,y2), (0,255,0), int(round(fh/150)), 8)
    return focv,fb


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

fp="opencv_face_detector.pbtxt"
fm="opencv_face_detector_uint8.pb"
ap="age_deploy.prototxt"
am="age_net.pdf"
gp="gender_deploy.prototxt"
gm="gender_net.pdf"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
al=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gl=['Male','Female']

fn=cv2.dnn.readNet(fm,fp)
an1=cv2.dnn.readNet(am,ap)
gn=cv2.dnn.readNet(gm,gp)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,fb=highlightFace(fn,frame)
    if not fb:
        print("No face detected")

    for faceBox in fb:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        gn.setInput(blob)
        genderPreds=gn.forward()
        gender=gl[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        an1.setInput(blob)
        agePreds=an1.forward()
        age=al[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

#To run this file: python gad.py --image <image name with extension>
