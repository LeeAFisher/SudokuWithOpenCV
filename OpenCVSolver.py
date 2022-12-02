import cv2
import numpy as np
import imutils
import pandas as pd

# Read image
#img = cv2.imread('sudoku2.jpg')
#cv2.imshow("Input image", img)

input_size = 50


def get_perspective(img, location, height = 900, width = 900):
    """Takes an image and location of an interesting region.
    And return the only selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imshow("Perspective", result)
    cv2.waitKey(0)
    return result


def filter_boxes(img):
    """Takes an image of a sudoku board and blacks out the grid lines"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    fill_color = [180,180,180]
    mask_value = 255  
    ret2, cleanboard = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(cleanboard, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    BoxContours = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4: 
            if cv2.contourArea(contour) > 3000:
                BoxContours.append(approx)
    color = [255, 255, 255]
    stencil  = np.zeros(img.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, BoxContours, color)
    sel      = stencil != mask_value # select everything that is not mask_value
    img[sel] = fill_color            # and fill it with fill_color
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(7,7),0)

    ret2, cleanboard = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow("wo",cleanboard)
    #cv2.waitKey(0)
    return cleanboard
    
def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg)
    cv2.waitKey(0)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 8, True)
        if len(approx) == 4:
            #print(cv2.contourArea(approx))
            location = approx
            break
    result = get_perspective(img, location)
    return result, location

# split the board into 81 individual images
def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255
            box = np.array(box).reshape(-1)
            boxes.append(box)
            #cv2.imshow("box", box)
            #cv2.waitKey(50)
    cv2.destroyAllWindows()
    return boxes


def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

# Read image
img = cv2.imread('sudoku1.jpg')


# extract board from input image
board, location = find_board(img)

#cv2.imshow("board", board)
#cv2.waitKey(0)
Gridless = filter_boxes(board)

#result = cleanboard.copy()
#cv2.imshow("wo", Gridless)
cv2.imshow("filtered", Gridless)
cv2.waitKey(0)

# print(gray.shape)
rois = split_boxes(Gridless)
np.shape(rois)

digits = pd.DataFrame(rois)
digits.to_csv("SudokuDigits.csv", index=None)

print(np.shape(rois))