import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from moviepy.editor import VideoFileClip
import time

class Lane_Finder:
    def __init__(self,
                 mtx=None, 
                 dist=None, 
                 calib_dir='camera_cal/',
                 nx=9,
                 ny=6,
                 objp=None,
                 src=None,
                 dst=None,
                 h_thresh=None,
                 l_thresh=(50, 255),
                 s_thresh=(120, 255), 
                 sx_thresh=(20, 100),
                 frame_buffer=5,
                 margin=100,
                 padding=200,
                 lane_length=49.0,
                 lane_width=3.7,
                 nwindows=9,
                 minpix=50,
                 output_dir='output_images/',
                 always_windows=False
                ):
        self.nx = nx
        self.ny = ny
        self.h_thresh = h_thresh
        self.l_thresh = l_thresh
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh
        self.frame_buffer = frame_buffer
        self.margin = margin
        self.padding = padding
        self.lane_length = lane_length
        self.lane_width = lane_width
        self.nwindows = nwindows
        self.minpix = minpix
        self.recent_fit = False
        self.output_dir = output_dir
        self.processing_video = False
        self.first = True
        self.always_windows = always_windows

        # Empty lists for polynomial fits in buffer
        self.left_fits = []
        self.right_fits = []
        self.left_fits_metric = []
        self.right_fits_metric = []

        # Generate 3D object points
        if objp is None:
            objp = np.zeros((ny*nx, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
            self.objp = objp
        # Try to calibrate camera if calibration information not provided
        if mtx is None:
            print('Calibrating camera with images in {}'.format(calib_dir))
            self.calibrate(calib_dir)
            self.calibrated = True

        # Load default src and dst points if none provided:
        if src is None:
            src = np.array([[208, 720], [595, 450],
                            [686, 450], [1102, 720]],
                           np.float32)
        if dst is None:
            dst = np.array([[208 + padding, 720], [208 + padding, 0],
                            [1102 + padding, 0], [1102 + padding, 720]],
                           np.float32)
    
        # Set up transformation matrix and inverse transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
              
    def calibrate(self, calib_dir):
        cal_images = os.listdir(calib_dir)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # termination criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for cal_image in cal_images:
            image_path = os.path.join(calib_dir, cal_image)
            img = plt.imread(image_path)
            img = np.copy(img)  # Create a writable copy
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret == True:
                objpoints.append(self.objp)
                corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(200)
            else:
                print('Corners not found for ', cal_image)
        cv2.destroyAllWindows()
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                           imgpoints, 
                                                           gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        print('Calibration successful!')
		
		
    def binary_lane_img(self, 
                        image, 
                        h_thresh=None,
                        l_thresh=None,
                        s_thresh=None, 
                        sx_thresh=None, 
                        diagnostic=False):
        if h_thresh == None:
            h_thresh = self.h_thresh
        if l_thresh == None:
            l_thresh = self.l_thresh
        if s_thresh == None:
            s_thresh = self.s_thresh
        if sx_thresh == None:
            sx_thresh = self.sx_thresh

        img = np.copy(image)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        if h_thresh is not None:
            h_binary = np.zeros_like(h_channel)
            h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        l_binary = np.zeros_like(l_channel)
        if l_thresh is not None:
            l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

        if diagnostic:
            color_binary = np.dstack((l_binary, sxbinary, s_binary)) * 255
            return color_binary
        else:
            binary = np.zeros_like(img[:, :, 0])
            if l_thresh is None:
                if h_thresh is None:
                    binary[(sxbinary == 1) | (s_binary == 1)] = 1
                else:
                    binary[((h_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
            else:
                if h_thresh is None:
                    binary[((l_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
                else:
                    binary[((h_binary == 1) & (l_binary == 1) & (s_binary == 1)) & (sxbinary == 1)] = 1
            return binary

    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        out_img = np.zeros((*binary_warped.shape, 3), dtype=np.uint8)
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(binary_warped.shape[0]//self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        return leftx, lefty, rightx, righty, out_img

    def fit_polynomial(self, binary_warped):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img, left_fit, right_fit

    def measure_curvature_pixels(self, left_fit, right_fit, ploty):
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        return left_curverad, right_curverad

    def process_image(self, image):
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        binary_img = self.binary_lane_img(undist)
        warped = cv2.warpPerspective(binary_img, self.M, (binary_img.shape[1], binary_img.shape[0]), flags=cv2.INTER_LINEAR)
        out_img, left_fit, right_fit = self.fit_polynomial(warped)

        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_curverad, right_curverad = self.measure_curvature_pixels(left_fit, right_fit, ploty)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = cv2.warpPerspective(color_warp, self.Minv, (image.shape[1], image.shape[0])) 
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        
        return result
"""
    def process_video(self, input_video_path, output_video_path):
        clip = VideoFileClip(input_video_path)
        white_clip = clip.fl_image(self.process_image)
        white_clip.write_videofile(output_video_path, audio=False)
"""
if __name__ == "__main__":
    src = np.float32([[208, 720], [595, 450], [686, 450], [1102, 720]])
    dst = np.float32([[300, 720], [300, 0], [980, 0], [980, 720]])
    lane_finder = Lane_Finder(src=src, dst=dst)
   # lane_finder.process_video("video01.mp4", "output_project_video.mp4")


 # Initialize the video capture object
    cap = cv2.VideoCapture('solidYellowLeft.mp4')

    # Variables for FPS calculation
    start_time = time.time()
    frames_count = 0

    # Loop to read and process frames from the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame using your lane detection algorithm
        processed_frame = lane_finder.process_image(frame)
        
        # Calculate FPS
        frames_count += 1
        elapsed_time = time.time() - start_time
        fps = frames_count / elapsed_time

        # Display FPS on the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow('frame', processed_frame)
        
        # Check for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()