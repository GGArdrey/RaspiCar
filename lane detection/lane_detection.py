import cv2
import numpy as np
import random

camwidth = 1920  
camheight = 1080
debug_lines = False
maxangle = 45


#----------------------------------------------------------
# stream

# creates a video capture
# performs lane detection on the video capture
# main loop for lane detection

def stream():
    cap = cv2.VideoCapture(2) #0-1 usually internal cam, 2-3 usually webcam if unsure run: "ls -la /dev/video*""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camwidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camheight)
    #region_mask = createmask(cap)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", camwidth, camheight)
    

    while(True):
        ret, frame = cap.read()
        img = frame
        #print(ret)
        if(not ret):
            print("No Frame, Check camera?")
            break
        try:
            img = line_detection(frame)
        except:    
            print("Error")
        cv2.imshow("frame",frame)
        cv2.imshow("final",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('lane detection/test.png', img)
            break
    cap.release()
    cv2.destroyAllWindows() 


#----------------------------------------------------------
# line_detection

# Uses the following methods in order to find and draw 
# lines onto the captured frame.
 
# Denoised - Denoises the picture 
 
# Gray - Converts picture to grayscale

# Canny - Canny Edge edge detection using opencv canny edge

# Masked - marks a target ROI (region of interest) 
# only shows values in that region

# Warped - Perspective warp to create a birds eye view of the ROI
def line_detection(frame):
    # A Kernel to denoise the frame
    kernel = np.ones((3,3), np.float32) / 9
    
    #Denoised frame after applying the kernel
    denoised = cv2.filter2D(frame, -1, kernel)
    
    #Grayscale Image
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)   
    cv2.imshow('gray',gray)
    
    #Edgedetection through canny edge
    canny = cv2.Canny(gray, 50, 150)
    cv2.imshow('canny', canny)
    
    #Region of interest mask applied
    masked = applymask(canny)
    cv2.imshow('masked', masked)
    
    #Perspective warping to gain a "skyview"
    warped = perspective_warp(masked)
    cv2.imshow('warped', warped)
    
    #Segmenting the regions that we are interested in
    segmented = segment_image(warped)
    
    #Hough Line detection
    detected_lines = hough_line_detect(frame, warped)
    
    #Line optimization (refining the results)
    clean_lines = line_opti(frame, detected_lines)
    
    frame = draw_lines(frame, clean_lines)
    
    
    return frame


#----------------------------------------------------------
# applymask

# creates a mask on the image that isolates a specific point
# of interest for our image

# here the parameters create a trapezoid at half the image height
# which widens towards the bottom
 
def applymask(frame):
    m_height, m_width = frame.shape 
    mask = np.zeros((m_height, m_width),dtype=np.uint8)
    
    trapezoid = np.array([
        [int(float(m_width * 0.25)), m_height],
        [int(float(m_width * 0.33)), int(float(m_height * 0.5))],
        [int(float(m_width * 0.66)), int(float(m_height * 0.5))],
        [int(float(m_width * 0.75)), m_height]
        ])
    
    cv2.fillPoly(mask, pts=[trapezoid], color=255)
    frame = cv2.bitwise_and(frame, mask)
    
    cv2.imwrite('lane detection/intermediates/masked.png', frame)
    return frame


#----------------------------------------------------------
# perspective_warp

# warps the region of interest to a full scale image
# creating a "birds eye view", to easier identify the lanes

def perspective_warp(img):
    p_height, p_width = img.shape
    
    src_points = np.float32([
        [int(p_width * 0.33), int(p_height * 0.5)],
        [int(p_width * 0.66), int(p_height * 0.5)],
        [int(p_width * 0.25), p_height],
        [int(p_width * 0.75), p_height]
        
    ])
    
    ratio_offset = 100
    
    warp_destination = np.float32([
        [ratio_offset, 0],
        [p_width - ratio_offset * 2, 0],
        [ratio_offset, p_height],
        [p_width - ratio_offset * 2, p_height],
    ])
    
    warp_mat = cv2.getPerspectiveTransform(src_points, warp_destination)
    
    warped = cv2.warpPerspective(img, warp_mat, (p_width,p_height))
    cv2.imwrite('lane detection/intermediates/Warped_Perspective.png', warped)
    return warped


#----------------------------------------------------------
# hough_line_detect

# performs hough line detection on the preprocessed frame
# returns lines that then will be postproceesed by other methods

def hough_line_detect(base_img, preprocessed):
    lines = cv2.HoughLines(preprocessed, 1, np.pi/180, 120)
    cvlines = []
    if lines is None:
        print("No lines found.")
        return cvlines
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
    
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
    
        # x0 stores the value rcos(theta)
        x0 = a*r
    
        # y0 stores the value rsin(theta)
        y0 = b*r
    
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
    
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
    
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
        cvlines.append([x1,y1,x2,y2])
        
    return cvlines


#----------------------------------------------------------
# segment_image

# builds a histogram to identify the regions where lines are
# present after warping
# returns those values for use in steering action

def segment_image(img):
    hist = np.sum(img, axis=0)
    
    mid = int(hist.shape[0]/2)
    
    base_L = np.argmax(hist[:mid])
    
    base_R = np.argmax(hist[mid:]) + mid
    
    bases = []
    
    bases.append(base_L)
    bases.append(base_R)
    return bases


#----------------------------------------------------------
# line_opti

# uses all detected lines to create only one line for the left
# and right lane respectively

def line_opti(frame, lines):
    seg_height, seg_width, _shape3 = frame.shape
    if lines is not None:
        opti_lines = []
        lines_L = []
        lines_R = []
        for x in lines:
            x1 = x[0]
            y1 = x[1]
            x2 = x[2]
            y2 = x[3]
            line_params = np.polyfit((x1, y1), (x2, y2), 1)
            
            k = line_params[0]
            d = line_params[1]
            
            if k < 0:
                lines_L.append((k,d))
            else:
                lines_R.append((k,d))
                
        if len(lines_L) > 0:
            avg_L = np.average(lines_L, axis=0)
            opti_lines.append(build_line(frame, avg_L))      
            
        if len(lines_R) > 0:
            avg_R = np.average(lines_R, axis=0)
            opti_lines.append(build_line(frame, avg_R))  
            
    return opti_lines  


#----------------------------------------------------------
# build_line

#builds line form equation format
    
def build_line(frame, params):
    b_height, shape2, shape3 = frame.shape
    k, d = params #slope, intercept
    
    if k == 0:
        k = 0.1
        
    y1 = b_height
    y2 = int(b_height * 0.5)
    x1 = int((y1 - d)/k)
    x2 = int((y2 - d)/k)
    
    return [x1, y1, x2, y2]
    

#----------------------------------------------------------
# draw_lines

#draws the lines onto the original frame
   
def draw_lines(img, lines):
    frame_mask = np.zeros_like(img)   
    #Line drawing on the original frame
    if(len(lines) > 0):
        for x in lines:
            cv2.line(frame_mask, (x[0], x[1]), (x[2], x[3]), (0, 255, 0),5)
            
    image__and_lines = cv2.addWeighted(img, 0.8, frame_mask,1 ,1)
            
    return image__and_lines
         

#----------------------------------------------------------
# single_img_detection

# performs line detection on a single picture
# uses the picture specified in the main function
    
def single_img_detection(img):
    frame = img
    kernel = np.ones((3,3), np.float32) / 9
    try:
        denoised = cv2.filter2D(frame, -1, kernel)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)   

    except:    
        print("Error")
        
    cv2.imwrite('lane detection/intermediates/gray.png', gray)
    canny = cv2.Canny(gray, 50, 150)
    cv2.imwrite('lane detection/intermediates/edges.png', canny)
    masked = applymask(canny)
    cv2.imwrite('lane detection/intermediates/masked.png', masked)
    warped = perspective_warp(masked)
    cv2.imwrite('lane detection/intermediates/warped.png', warped)
    segmented = segment_image(warped)
    detected_lines = hough_line_detect(frame, warped)
    clean_lines = line_opti(frame, detected_lines)    
    frame = draw_lines(frame, clean_lines)

    cv2.destroyAllWindows() 
    cv2.imwrite('lane detection/intermediates/linesDetected.png', frame)

#----------------------------------------------------------
# single_image_capture
# 
# Opens the camera
# Takes picture when you press the "q" key
# must have the "frame" window as active window to work

def single_image_capture():
    cap = cv2.VideoCapture(2) #0-1 usually internal cam, 2-3 usually webcam if unsure run: "ls -la /dev/video*""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camwidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camheight)
    
    while(True):
        ret, frame = cap.read()
        if(not ret):
            print("No Frame.")
            break
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('lane detection/Single_Image.png', frame)
            break
    cap.release()
    
    
if __name__ == "__main__":
    #stream()
    img = cv2.imread("lane detection/Single_Image.png")
    single_img_detection(img)
    #single_image_capture()