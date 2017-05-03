import cv2
import os
import numpy as np
import sys

root = '/home/lferrer/Downloads/Full Res Videos/'
resX = 1920
resY = 1080
padding = 0.20
cropsRoot = '/home/lferrer/Documents/Synthetic/Third Person' 
folder = ''
if len(sys.argv) == 1:
	print "ERROR: The scene-character folder must be specified as an argument."
	#folder = 'Ice Natural'
	exit()
else:
	folder = sys.argv[1]

# Define the root folders 
thirdPersonRoot = root + 'Third Person/' + folder
maskRoot = root + 'Masks/' + folder
cropsRoot = cropsRoot + '/' + folder

if not(os.path.exists(cropsRoot)):
	os.mkdir(cropsRoot)

# Return all files in thirdPersonRoot
for videoName in next(os.walk(thirdPersonRoot))[2]:
	tpVideoName = thirdPersonRoot + '/' + videoName	
	maskVideoName = maskRoot + '/' + videoName
	if not(os.path.exists(maskVideoName)):
			print "ERROR: File: " + maskVideoName + " not found"
	else:
		angleFolder = cropsRoot + '/' + videoName[:-4]
		print angleFolder
		if not(os.path.exists(angleFolder)):
			os.mkdir(angleFolder)
		tpVideo = cv2.VideoCapture(tpVideoName)
		maskVideo = cv2.VideoCapture(maskVideoName)
		frame = 0
		retTP, image = tpVideo.read()
		retMask, blackImg = maskVideo.read()
		while(retTP and retMask):
			cropFilename = angleFolder + '/' + str(frame) + ".jpg"
			if not(os.path.exists(cropFilename)):
				blackImg = cv2.cvtColor(blackImg, cv2.COLOR_BGR2GRAY)
				ret, blackImg = cv2.threshold(blackImg, 127, 255, cv2.THRESH_BINARY)	
				minX = resX
				minY = resY
				maxX = 0
				maxY = 0
				locs = cv2.findNonZero(blackImg)
				if locs is not None and len(locs) > 0:
					for l in locs:
						loc = l[0]
						if loc[0] < minX:
							minX = loc[0]
						if loc[0] > maxX:
							maxX = loc[0]
						if loc[1] < minY:
							minY = loc[1]
						if loc[1] > maxY:
							maxY = loc[1]
					width = maxX - minX
					minX = minX - width * padding
					maxX = maxX + width * padding
					height = maxY - minY
					minY = minY - height * padding
					maxY = maxY + height * padding
					minX = int(max(minX, 0))
					minY = int(max(minY, 0))
					maxX = int(min(maxX, resX))
					maxY = int(min(maxY, resY))				
					cropped = image[minY:maxY, minX:maxX]				
					cv2.imwrite(cropFilename, cropped)
			frame = frame + 1
			retTP, image = tpVideo.read()
			retMask, blackImg = maskVideo.read()
		tpVideo.release()
		maskVideo.release()