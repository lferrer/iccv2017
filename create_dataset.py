import numpy as np
from skimage.measure import compare_ssim as ssim
import lmdb
import cv2
import caffe
from os import path
from random import randint

# Config section of the script
POSITIVE_TRAIN_SAMPLES = 7#140000
NEGATIVE_TRAIN_SAMPLES = 7#140000
POSITIVE_VAL_SAMPLES = 1#20000
NEGATIVE_VAL_SAMPLES = 1#20000
POSITIVE_TEST_SAMPLES = 2#40000
NEGATIVE_TEST_SAMPLES = 2#40000
INTER_VIDEO_RATE = 0.80
VIDEO_LENGTH = 4292
SCENARIOS = ['Fire Pit', 'Grass Standard', 'Ice Natural', 'Reflection Agent', 'Temple Barbarous']
ELEVATION_ANGLES = [330, 340, 350]
ROTATION_ANGLES = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, \
                   200, 220, 240, 260, 280, 300, 320, 340]
DS_ROOT = '/home/lferrer/Documents/Synthetic'
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
INTRA_VIDEO_SSIM_THRESHOLD = 0.5
# Video frames were to 128 by 171 (According to original paper)
FRAME_WIDTH = 128
FRAME_HEIGHT = 171

# Helper functions
# Returns the filename of a given image based on the dataset structure
def buildFilename(firstPerson, scenario, index, elevation = 330, rotation = 0):
    imageName = str(index) + '.jpg'
    if firstPerson:
        filename = path.join(DS_ROOT, 'First Person', scenario, imageName)
    else:
        angleFolder = str(elevation) + '-' + str(rotation)
        filename = path.join(DS_ROOT, 'Third Person', scenario, angleFolder, imageName)
    return filename

# Returns a random element from a list
def getRndEl(myList):
    return myList[randint(0, len(myList) - 1)]

# Returns a random positive image tuple
def getRndPosImg():
    scenario = getRndEl(SCENARIOS)
    elevation = getRndEl(ELEVATION_ANGLES)
    rotation = getRndEl(ROTATION_ANGLES)
    index = randint(1, VIDEO_LENGTH)
    fpFilename = buildFilename(True, scenario, index)
    tpFilename = buildFilename(False, scenario, index, elevation, rotation)
    return fpFilename, tpFilename

# Returns a random negative inter-video image tuple
def getRndNegInterImg():
    scenario = getRndEl(SCENARIOS)
    index = randint(1, VIDEO_LENGTH)
    fpFilename = buildFilename(True, scenario, index)
    scenario = getRndEl(SCENARIOS)
    index = randint(1, VIDEO_LENGTH)
    elevation = getRndEl(ELEVATION_ANGLES)
    rotation = getRndEl(ROTATION_ANGLES)
    tpFilename = buildFilename(False, scenario, index, elevation, rotation)
    return fpFilename, tpFilename

# Returns a random negative intra-video image tuple
def getRndNegIntraImg():
    scenario = getRndEl(SCENARIOS)
    elevation = getRndEl(ELEVATION_ANGLES)
    rotation = getRndEl(ROTATION_ANGLES)
    index = randint(1, VIDEO_LENGTH)
    fpFilename = buildFilename(True, scenario, index)    
    index = randint(1, VIDEO_LENGTH)
    tpFilename = buildFilename(False, scenario, index, elevation, rotation)
    fpImage = cv2.imread(fpFilename)
    tpImage = cv2.imread(tpFilename)
    fpImage = cv2.resize(fpImage, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    tpImage = cv2.resize(tpImage, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    dif = ssim(fpImage, tpImage, multichannel=True)
    while dif > INTRA_VIDEO_SSIM_THRESHOLD:
        index = randint(1, VIDEO_LENGTH)
        tpFilename = buildFilename(False, scenario, index, elevation, rotation)
        tpImage = cv2.imread(tpFilename)
        tpImage = cv2.resize(tpImage, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
        dif = ssim(fpImage, tpImage, multichannel=True)
    return fpFilename, tpFilename


def loadImageTuple(fpFilename, tpFilename):
    # Load the images from disk
    fpImage = cv2.imread(fpFilename)
    tpImage = cv2.imread(tpFilename)

    
    # INTER_AREA is the best for image decimation (According to OpenCV documentation)
    fpImage = cv2.resize(fpImage, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)
    tpImage = cv2.resize(tpImage, (FRAME_WIDTH, FRAME_HEIGHT), cv2.INTER_AREA)

    # Stack the images for datum insertion
    imageData = np.dstack((fpImage, tpImage))
    return imageData

def addToDB(txn, imageData, label, i):
    # Build the caffe Datum
    datum = caffe.proto.caffe_pb2.Datum()
    datum.height = imageData.shape[0]
    datum.width = imageData.shape[1]
    #Concat the two images using 6 channels
    datum.channels = imageData.shape[2]
    datum.data = imageData.tobytes()
    datum.label = label
    str_id = '{:08}'.format(i)

    # The encode is only essential in Python 3
    txn.put(str_id.encode('ascii'), datum.SerializeToString())

def generateSamples(txn, nSamples, label, getRndImgFunc):
    for i in range(nSamples):
        # Get a random image
        fpFilename, tpFilename = getRndImgFunc()
        imgTuple = fpFilename + '-' + tpFilename
        while imgTuple in hitDict:
            # Get an unused random image
            fpFilename, tpFilename = getRndImgFunc()
            imgTuple = fpFilename + '-' + tpFilename

        # Add the random image to the used list
        hitDict[imgTuple] = True

        # Get the images from the disk
        imageData = loadImageTuple(fpFilename, tpFilename)

        # Add to the lmdb
        addToDB(txn, imageData, label, i)

def createDB(name, nPosSmaples, nNegSamples):
    env = lmdb.open(name, map_size=map_size)
    with env.begin(write=True) as txn:
        # Generate positive samples
        generateSamples(txn, nPosSmaples, POSITIVE_LABEL, getRndPosImg)
        
        # Generate negative samples
        interVideoSamples = int(nNegSamples * INTER_VIDEO_RATE)
        intraVideoSamples = nNegSamples - interVideoSamples

        generateSamples(txn, interVideoSamples, NEGATIVE_LABEL, getRndNegInterImg)
        generateSamples(txn, intraVideoSamples, NEGATIVE_LABEL, getRndNegIntraImg)

# Prepare the DBs
hitDict = {}

# Creating the map size. According to the LMDB website: On 64-bit there is no penalty for making this huge (say 1TB). 
map_size = 1e12

# Create DBs
createDB('train', POSITIVE_TRAIN_SAMPLES, NEGATIVE_TRAIN_SAMPLES)
createDB('val', POSITIVE_VAL_SAMPLES, NEGATIVE_VAL_SAMPLES)
createDB('test', POSITIVE_TEST_SAMPLES, NEGATIVE_TEST_SAMPLES)
        
exit()