# -*- coding: utf-8 -*-
"""
Running this file will generate a big data set of 50 x 50 
black and white images of different digits in different fonts.

@author: Lee
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from numpy.random import default_rng
import pandas as pd
from itertools import product

#let's go digit by digit
rng = default_rng()


textfile = open('font-names.txt', "r")
fonts = textfile.readlines()
for n in range(len(fonts)):
    fonts[n] = "C:\Windows\Fonts\\" + fonts[n]
    fonts[n] = fonts[n].rstrip(fonts[n][-1])

digits = ["1","2","3","4","5","6","7","8","9"]
xoffsets = [13,17]
yoffsets = [8,12]
sizes = [28,36,48]

mid28 = [16,19]
mid36 = [8,3]
mid48 = [4,-6]

mids = [[17,10],[9,4], [5,-5]]
shifts = [np.arange(-6,7,2), np.arange(-4,5,2), np.arange(-2,3,2)]

DataArray = np.zeros((2501,1))
fontindices = np.arange(60,80)
testImg = Image.new(mode = "L", size = (50,50), color = (0) )
draw = ImageDraw.Draw(testImg)
draw.text((17-9,10-9), digits[3], fill = (255), font = ImageFont.truetype(fonts[0], 36), )
testImg.show()

for n in fontindices:
    print(n)
    for s in [0, 1, 2]:
        print(s)
        for j, k in product(shifts[s],shifts[s]):
            for d in range(len(digits)):
                testImg = Image.new(mode="L", size=(50, 50), color=(0))
                draw = ImageDraw.Draw(testImg)
                draw.text((mids[s][0]+j,mids[s][1] + k), digits[d], fill=(
                255), font=ImageFont.truetype(fonts[n], sizes[s]), )
                testImg = testImg.filter(ImageFilter.GaussianBlur(1))
                testImg = testImg.point( lambda p: 255 if p > 100 else 0 )
                imgArray1 = np.array(testImg).reshape(-1, 1)
                imgArray1 = np.concatenate(([[d+1]], imgArray1))
                DataArray = np.concatenate(
                            (DataArray, imgArray1), axis=1)
            DataArray = np.concatenate((DataArray, np.zeros((2501, 8))), axis=1)
DataArray = DataArray.T
TrainingDigits = pd.DataFrame(DataArray)
TrainingDigits.to_csv("TrainingDigits.csv", index=None)