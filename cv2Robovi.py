from curses import echo
from http.client import IM_USED
from math import floor
import os
from re import S
import sys
from time import sleep
import time
import cv2
# from cv2 import sqrt
import numpy as np
# import matplotlib.pyplot as plt
import tkinter as tk
# import PIL
from tkinter import *
from matplotlib import pyplot as plt
roberts_cross_v = np.array([[1, 0], [0, -1]])

roberts_cross_h = np.array([[0, 1], [-1, 0]])


img2 = cv2.imread('Uvod/tijaImage.png', 1)
img = cv2.imread('Uvod/natureSpot.png', 1)
img2RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2RGB = cv2.cvtColor(img2RGB, cv2.COLOR_BGR2GRAY)

# img = cv2.imread('Uvod/test.png', 1)

# cv2.imshow("slikca", img)
# cv2.waitKey(0)

# https://dsp.stackexchange.com/questions/898/roberts-edge-detector-how-to-use


def my_roberts(slika):
    # img = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    vertical = cv2.filter2D(slika, -1, roberts_cross_v)
    horizontal = cv2.filter2D(slika, -1, roberts_cross_h)

    combine = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0)
    return combine * 3


def my_prewitt(slika):

    # img = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    # img = cv2.bilateralFilter(img, 9, 75, 75) #za blurr
    # tta je obratna ? in zamenjat bi jih rabu
    y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # sm jih zamenju

    img_x = cv2.filter2D(slika, -1, x)
    img_y = cv2.filter2D(slika, -1, y)

    combine = np.concatenate((img_x, img_y), axis=1)
    return combine * 3


def my_sobel(slika):
    # slike = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y = np.array([[1, 2, 3], [0, 0, 0], [-1, -2, -1]])
    # uporaba ze obstojece funkcije
    # img_x2 = cv2.Sobel(slika, cv2.CV_8U, 1, 0, ksize=5)
    # img_y2 = cv2.Sobel(slika, cv2.CV_8U, 0, 1, ksize=5)

    img_x = cv2.filter2D(slika, -1, x)
    img_y = cv2.filter2D(slika, -1, y)
    combine = img_x + img_y  # mby uporab cv2.add()
    # combine2 = img_x2 + img_y2
    # combine = np.concatenate((combine, combine2), axis=1)
    return combine


def canny2(slika, sp_prag, zg_prag):
    lower = sp_prag
    upper = zg_prag

    Cannyimage = cv2.Canny(slika, lower, upper)

    return Cannyimage


def spremeni_kontrast(slika, alfa, beta):
    print("alfa je ", alfa)
    slika = cv2.multiply(slika, alfa)
    slika = cv2.add(slika, beta)
    return slika


tink = tk.Tk()
tink.title("Adjust settings")
tink.configure(width=500, height=300)
tink.configure(bg='lightgray')
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
