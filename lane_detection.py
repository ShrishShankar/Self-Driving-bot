import cv2
import io
import numpy as np
# import matplotlib.pyplot as plt
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import RPi.GPIO as GPIO

#disabling GPIO warnings
GPIO.setwarnings(False)

#GPIO setup for raspberry pi
GPIO.setmode(GPIO.BOARD)    
GPIO.setup(31, GPIO.OUT)
GPIO.setup(33, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)
GPIO.setup(37, GPIO.OUT)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1648, 1232)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1648, 1232))
# live_vid = cv2.VideoCapture(0)                  # enter camera number, mostly going to be 0
#
# # Check if camera opened successfully
# if not live_vid.isOpened():
#     print("Error opening video")

# allow the camera to warmup
time.sleep(0.1)
print('start your engine')
# Read until video is completed
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    image = frame.array
    print("ready")
    #cv2.imshow('normal', image)
    # ret, frame = live_vid.read()  # ret - returns True/False based if video is playing or not, frame - returns the frame
    # being played


    # Lane detection code

    # pre-processing for canny
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # canny
    threshold_1 = 120
    threshold_2 = 200

    canny_image = cv2.Canny(image_gray, threshold_1, threshold_2)
    # plt.imshow(canny_image, cmap='gray')
    # plt.show()
    # plt.waitforbuttonpress(0)
    #print('canny')
    #cv2.imshow('canny', canny_image)
    #key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    #rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    #if key == ord("q"):
     #   break
    
    # Region of interest
    height, width = canny_image.shape
    # print(height)
    # print(width)

    vertices = np.array([[(226, 1230), (280, 1124), (1366, 1124), (1369, 1227)]], dtype=np.int32)
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(canny_image, mask)
    # print(masked_image)
    # plt.imshow(masked_image, cmap='gray')
    # plt.waitforbuttonpress(0)

    # Getting perspective -> Bird's eye view
    source_vertices = np.array([[226, 1230], [280, 1124], [1366, 1124], [1369, 1227]], dtype=np.float32)
    destination_vertices = np.array([[0, height], [0, 0], [width, 0], [width, height]], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(source_vertices, destination_vertices)
    perspective_image = cv2.warpPerspective(masked_image, perspective_matrix, dsize=(width, height))
    # plt.imshow(perspective_image, cmap='gray')
    # plt.waitforbuttonpress(0)

    # Hough Transformation
    rho = 2             # distance resolution in pixels
    theta = np.pi/180   # angular resolution in radians
    threshold = 40      # minimum number of votes
    min_line_len = 100  # minimum number of pixels making up a line
    max_line_gap = 50   # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(perspective_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # plt.imshow(lines)
    # plt.waitforbuttonpress(0)

    # Create an empty black image
    line_image = np.zeros((perspective_image.shape[0], perspective_image.shape[1], 1), dtype=np.uint8)
    x1l_vals = []
    x1r_vals = []
    x2l_vals = []
    x2r_vals = []
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # print('x1:{}, y1:{}, x2:{}, y2:{}'.format(x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), [255, 0, 0], 20)
                # line_image[0, 1241] = 255
                # line_image[881, 1318] = 255
                if x1 and x2 < 400:
                    x1l_vals.append(x1)
                    x2l_vals.append(x2)
                if x1 and x2 > 700:
                    x1r_vals.append(x1)
                    x2r_vals.append(x2)
    except:
        print('no lines detected')
        rawCapture.truncate(0)
        continue
    # print(lines.size)
    # print('max, min x1:{}, {} '.format(max(x1_vals), min(x1_vals)))
    # print('max, min y1:{}, {} '.format(max(y1_vals), min(y1_vals)))
    # print('max, min x2:{}, {} '.format(max(x2_vals), min(x2_vals)))
    # print('max, min y2:{}, {} '.format(max(y2_vals), min(y2_vals)))

    if len(x1l_vals) == 0:
        print('Zero Value Error: 1L')
        rawCapture.truncate(0)
        continue
    else:
        x1l_avg = sum(x1l_vals)//len(x1l_vals)
    if len(x2l_vals) == 0:
        print('Zero Value Error: 2L')
        rawCapture.truncate(0)
        continue
    else:
        x2l_avg = sum(x2l_vals)//len(x2l_vals)
    if len(x1r_vals) == 0:
        print('Zero Value Error: 1R')
        rawCapture.truncate(0)
        continue
    else:
        x1r_avg = sum(x1r_vals)//len(x1r_vals)
    if len(x2r_vals) == 0:
        print('Zero Value Error: 2R')
        rawCapture.truncate(0)
        continue
    else:
        x2r_avg = sum(x2r_vals)//len(x2r_vals)

    x1_avg = (x1l_avg + x1r_avg)//2
    x2_avg = (x2l_avg + x2r_avg)//2

    # print(x1_avg)
    # print(x2_avg)

    cv2.line(line_image, (x1_avg, 0), (x2_avg, 882), [255, 0, 0], 10)

    α = 1
    β = 1
    γ = 0

    # determining orientation
    orientation = width//2 - (x1_avg+x2_avg)//2
    print(orientation)
    
    # Resultant weighted image is calculated as follows: original_img * α + img * β + γ
    Image_with_lines = cv2.addWeighted(perspective_image, α, line_image, β, γ)
    cv2.imshow('Image_with_lines is LIVE!!', Image_with_lines)
    cv2.putText(Image_with_lines, 'Result = {}'.format(orientation), org=(100, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(255, 0, 0),
                thickness=2)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    #motor control messages to arduuino from raspberry pi
    if (orientation == 0):  #forward, decimal=0
        GPIO.output(31, False)
        GPIO.output(33, False)
        GPIO.output(35, False)
        GPIO.output(37, False)
        print('Forward')
    elif (orientation > 0 and orientation <= 10):   #right1, decimal=1
        GPIO.output(31, True)
        GPIO.output(33, False)
        GPIO.output(35, False)
        GPIO.output(37, False)
        print('right1')
    elif (orientation > 10 and orientation <= 20):  #right2, decimal=2
        GPIO.output(31, False)
        GPIO.output(33, True)
        GPIO.output(35, False)
        GPIO.output(37, False)
        print('right2')
    elif (orientation > 20):    #right3, decimal=3
        GPIO.output(31, True)
        GPIO.output(33, True)
        GPIO.output(35, False)
        GPIO.output(37, False)
        print('right3')
    elif (orientation < 0 and orientation >= -10):   #left1, decimal=4
        GPIO.output(31, False)
        GPIO.output(33, False)
        GPIO.output(35, True)
        GPIO.output(37, False)
        print('left1')
    elif (orientation < -10 and orientation >= -20):    #left2,decimal=5
        GPIO.output(31, True)
        GPIO.output(33, False)
        GPIO.output(35, True)
        GPIO.output(37, False)
        print('left2')
    elif (orientation < -20):   #left3, decimal=6
        GPIO.output(31, False)
        GPIO.output(33, True)
        GPIO.output(35, True)
        GPIO.output(37, False)
        print('left3')

    # clear the stream in preparation for the next frame
    #rawCapture.truncate(0)
    # # Press 'enter' on keyboard to exit
    #if cv2.waitKey(1) == 13:
     #   break
    #rawCapture.truncate(0)
    # # Break the loop
    # else:
    #     print("Error opening video")
    #     break

# # the video capture object
# live_vid.release()  # closes video file or video capturing device

# Closes all the frames
cv2.destroyAllWindows()
