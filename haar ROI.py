import cv2
import os
from os.path import realpath, normpath
index = 0
h = normpath(realpath(cv2.__file__) + '/../data/')
print(h)
mouth_cascade = cv2.CascadeClassifier(h+'/haarcascade_mouth.xml')
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')
for label in range(0,10):
    j = 1
    for i in range(0, 7):
        print('KIM_%d_0%d.mp4' % (label,i))
        cap = cv2.VideoCapture('Dataset/KIM_%d_0%d.mp4' % (label,i))
        # recommend avi file, mp4 file often cause some error
        ret, frame = cap.read()
        ds_factor = 0.5
        height, width = frame.shape[:2]

        frame = cv2.resize(frame, (width, height), fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 30)

        for (x, y, w, h) in mouth_rects:
            #y = y-round(h*0.75)
            m = cv2.rectangle(frame, (x, y-round(h/4)), (x + w, y+round(h*3/4)), (0, 255, 0), 3)
            break
        track_window = (x, y, w, h)
        roi = frame[y-round(h/4):y+round(h*3/4), x:x + w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret2, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res = cv2.bitwise_and(roi, roi, mask=mask)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 255], [0, 180, 0, 255])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        while (1):
            ret, frame = cap.read()
            print(ret)
            if not os.path.exists('Dataset/News/S%d/%d' % (label, i)):
                os.makedirs('Dataset/News/S%d/%d'% (label, i))
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)
                # apply meanshift to get the new location
                #ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                # Draw it on image
                x, y, w, h = track_window
                img2 = cv2.rectangle(frame, (x, y-round(h/4)), (x + w, y+round(h*3/4)), 0, 1)
                print(j)
                lip = frame[y-round(h/4): + y+round(h*3/4), x:x + w]
                video = cv2.resize(lip, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("Dataset/News/S%d/%d/%d.jpg" % (label,i, j), video)
                j = j +1
                k = cv2.waitKey(60) & 0xff
                if k == 27:
                    break
                else:
                    continue

            else:
                print(f"Nombre d'images de la video -> {j}")
                break

cv2.waitKey(27)
cv2.destroyAllWindows()
cap.release()
