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
