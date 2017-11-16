import os
from PIL import Image
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from scipy.misc import imsave

# flipping images
# for root, dirs, files in os.walk('./data/non-vehicles/', topdown=True):
# 	    for fn in files:
# 	    	if not 'DS_Store' in fn:
# 	    		img = Image.open(os.path.join(root,fn)).transpose(Image.FLIP_LEFT_RIGHT)
# 	    		img.save('./data/non-vehicles/Extra_flipped/'+fn)

counter = 0

def run_image(img):

	global counter

	scales = (0.68,0.75,0.83,1,1.25,1.55,1.95,2.44)

	for s in scales:
		img_patch = img[np.int(578-s*32):np.int(578+s*31),np.int(496-s*32):np.int(496+s*31)]
		img_64 = cv2.resize(img_patch, (64, 64))
		imsave('./data/non-vehicles/Extra_line/image_2_'+str(counter)+'.png',img_64)
		counter += 1
	#print(counter)
	return img

def run_video(input, output):
    # run video
    video = VideoFileClip(input)
    out_video = video.fl_image(run_image)
    out_video.write_videofile(output, audio=False)

if __name__ == '__main__':
    
    run_video('short_begin.mp4', 'out.mp4')