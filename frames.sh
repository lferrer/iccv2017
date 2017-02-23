#This file traverses the UCF-101 folder structure and extracts the frames with FFMPEG
find '/home/lferrer/Downloads/UCF-101' -type f -exec ffmpeg -i {} {}%04d.jpg \; -exec rm {} \;