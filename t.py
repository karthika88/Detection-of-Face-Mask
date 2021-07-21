from scipy.spatial import distance
import torch
import os
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',classes=1, autoshape=False)
model.load_state_dict(torch.load('runs/train/exp/weights/best.pt')['model'].state_dict())
model = model.fuse().autoshape()
cap=cv2.VideoCapture(0)
while True:
	ret,img=cap.read()
	if not ret:
		break
	results = model(img)
	df=results.pandas().xyxy[0]
	print(df.head())
	if df.shape[0]>0:
		for i in range(df.shape[0]):
			if df.confidence[i]>0.5:
				startpoint=(int(df.xmin[i]),int(df.ymin[i]))
				endpoint=(int(df.xmax[i]),int(df.ymax[i]))
				color=[0,0,255]
				thickness=2
				image=img
				text= str(df.confidence[i])
				cv2.rectangle(image, startpoint, endpoint, color, thickness)
				cv2.putText(image,text,(50, 50),cv2.FONT_HERSHEY_SIMPLEX , 1, color,
				thickness, cv2.LINE_AA)
				cv2.imshow('img',image)
	else:
		cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
cv2.destroyAllWindows()
