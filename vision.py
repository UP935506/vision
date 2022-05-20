import cv2
import pytesseract
import pyttsx3
import time
from PIL import Image, ImageFilter
import numpy as np
from email.headerregistry import Address


def threshold(image, thresh1 = 127, thresh2 = 255):
    T_, thresholded = cv2.threshold(image, thresh1, thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def closer(image, kernel_size = 20, iterations = 50):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations
    )
    return closed 

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurr = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpen = float(amount + 1) * image - float(amount) * blurr
    sharpen = np.maximum(sharpen, np.zeros(sharpen.shape))
    sharpen = np.minimum(sharpen, 255 * np.ones(sharpen.shape))
    sharpen = sharpen.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurr) < threshold
        np.copyto(sharpen, image, where=low_contrast_mask)
    return sharpen

address = "http://10.128.9.92:8080/video"

capture = cv2.VideoCapture(0)
while True:
    ret_val, img = capture.read()   
    img = unsharp_mask(img)
    imgPP = closer(img)
    imgPP = threshold(imgPP)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_and(imgPP, img)

    cv2.imshow('Video', img)
    
    txt = pytesseract.image_to_string(img)
    if txt:print(txt)
     
    engine = pyttsx3.init()
    engine.say(txt)
    engine.runAndWait()
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break 
capture.release()
