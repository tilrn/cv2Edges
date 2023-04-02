from curses import echo
from http.client import IM_USED
from math import floor
from re import S
from time import sleep
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from matplotlib import pyplot as plt
roberts_cross_v = np.array([[1, 0], [0, -1]])

roberts_cross_h = np.array([[0, 1], [-1, 0]])


img2 = cv2.imread('slike/tijaImage.png', 1)
img = cv2.imread('slike/natureSpot.png', 1)
img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2RGB = cv2.cvtColor(img2RGB, cv2.COLOR_BGR2GRAY)
# img2RGB = cv2.GaussianBlur(img2RGB, (5, 5), 0)


def my_roberts(slika):
    # img = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    vertical = cv2.filter2D(slika, -1, roberts_cross_v)
    horizontal = cv2.filter2D(slika, -1, roberts_cross_h)

    combine = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0)
    return combine * 3


def my_prewitt(slika):

    y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # sm jih zamenju

    img_x = cv2.filter2D(slika, -1, x)
    img_y = cv2.filter2D(slika, -1, y)
    # to display x any y edge detection
    combine = np.concatenate((img_x, img_y), axis=1)
    return combine * 3


def my_sobel(slika):

    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 3], [0, 0, 0], [-1, -2, -1]])

    img_x = cv2.filter2D(slika, -1, x)
    img_y = cv2.filter2D(slika, -1, y)
    combine = cv2.add(img_x, img_y)
    return combine


def canny2(slika, sp_prag, zg_prag):
    lower = sp_prag
    upper = zg_prag

    Cannyimage = cv2.Canny(slika, lower, upper)

    return Cannyimage

# changes contrast alpha = contrast, beta = brightness


def spremeni_kontrast(slika, alfa, beta):
    print("alfa je ", alfa)
    print("beta je ", beta)
    slika = cv2.multiply(slika, alfa)
    slika = cv2.add(slika, beta)
    return slika


def get():
    brightnes2 = brightnes.get()
    contrast2 = contrast.get()
    brightnes3 = int(brightnes2)
    contrast3 = int(contrast2)
    contrastImage = spremeni_kontrast(img2RGB, contrast3 * 0.2, brightnes3)

    # my_Roberts
    myRoberts = my_roberts(contrastImage)
    myRobertsNormal = my_roberts(img2RGB)
    # my_prewitt
    myPrewitt = my_prewitt(contrastImage)
    myPrewittNormal = my_prewitt(img2RGB)
    # my_sobel
    sobel = my_sobel(contrastImage)
    sobelNormal = my_sobel(img2RGB)
    # canny
    canny = canny2(contrastImage, 10, 10)
    cannyNormal = canny2(img2, 100, 100)

    # izpisi
    cv2.imshow("Normal Roberts", myRobertsNormal)
    cv2.imshow("tempered Roberts", myRoberts)
    # cv2.imshow("Normal Prewitt", myPrewittNormal)
    # cv2.imshow("tempered Prewitt", myPrewitt)

    # cv2.imshow("Normal Sobel", sobelNormal)
    # cv2.imshow("Tempered Sobel", sobel)

    # cv2.imshow("Normal Canny", cannyNormal)
    # cv2.imshow("Tempered Canny", canny)
    cv2.waitKey(0)


tink = tk.Tk()
tink.title("Adjust settings")
tink.configure(width=400, height=300)
tink.configure(bg='dark')
brightnes = DoubleVar()
contrast = DoubleVar()
scale = Scale(tink, from_=100, to=-100, variable=brightnes,
              orient=HORIZONTAL, label="Brightnes")
scale2 = Scale(tink, from_=20, to=0, variable=contrast,
               orient=HORIZONTAL, label="Contrast")

button = tk.Button(tink, text="send", width=20, command=get)
scale.pack()
scale2.pack()
button.pack()
tink.mainloop()
