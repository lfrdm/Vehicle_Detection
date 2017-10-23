import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

#img = mpimg.imread('./test_images/test5.jpg')
#draw_image = np.copy(image)
#img = img.astype(np.float32)/255
# define heatmap
#heat = np.zeros_like(img[:,:,0]).astype(np.float)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hist_bins):
    
    draw_img = np.copy(img)
    # boxlist for this scale
    box_list = []

    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_features = X_scaler.transform(np.hstack((hist_features,hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                box = ((xbox_left,ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                box_list.append(box)

    return box_list

# Define a class for running average of bounding boxes
class AverageBB():
    def __init__(self):
        # list of bounding boxes
        self.bb_list = []


    def set(self, bb):
        # add last bb to list
        self.bb_list.append(bb)
       
        # always keep last 5 bb
        if(len(self.bb_list)>5):
            self.bb_list.pop(0)

    def get(self):
        # get the average of the bb
        bb_out = []
        for bb in self.bb_list:
            bb_out += bb

        return bb_out

dist_pickle = pickle.load( open("svm.pckl", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
#spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

averageBB = AverageBB()

def run_image(img):

    global svc, X_scaler, orient, pix_per_cell, cell_per_block, hist_bins, averageBB

    # scale img
    img = img.astype(np.float32)/255

    # init heatmap with zeros
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # define y-region band
    ystart = 400
    ystop = 656
    
    # scale series
    scale = (0.75, 1.0, 1.25, 1.5, 1.75, 2)
    
    # init mutli scale box list
    ms_box_list = []

    for s in scale:
        box_list = find_cars(img, ystart, ystop, s, svc, X_scaler, orient, pix_per_cell, cell_per_block, hist_bins)
        ms_box_list += box_list

    averageBB.set(ms_box_list)

    # Add heat to each box in box list
    heat = add_heat(heat,averageBB.get())
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,20)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img*255

def run_video(input, output):
    # run video
    video = VideoFileClip(input)
    out_video = video.fl_image(run_image)
    out_video.write_videofile(output, audio=False)

if __name__ == '__main__':
    
    #run_video('project_video.mp4', 'out.mp4')
    img = mpimg.imread('./test_images/test5.jpg')
    img = img.astype(np.float32)/255

    out_img = run_image(img)

    plt.imshow(out_img)
    plt.show()












