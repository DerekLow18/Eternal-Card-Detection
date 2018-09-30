'''
A script to extract the name of a card from a digital client.
To do this, we will first scan the entire page for cards,
then extract the name of the cards via image processing.
'''

import cv2
import numpy as np
import pyscreenshot as ImageGrab
import time


def grab_card(printscreen):
    gray = cv2.cvtColor(printscreen,cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, threshold1 = 100, threshold2 = 100)
    return gray


#record the screen
def main():
    #1024 x 768 windowed mode
    last_time = time.time()
    while(True):
        printscreen = np.array(ImageGrab.grab(bbox=(0,40,1024,768)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        thresh = grab_card(printscreen)
        
        #--- find contours on the result above ---
        (_, contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #--- since there were few small contours found, retain those above a certain area ---
        im2 = printscreen.copy()
        count = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                count+=1
                cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)
        cv2.imshow('window',thresh)
        print('There are {} cards'.format(count))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()