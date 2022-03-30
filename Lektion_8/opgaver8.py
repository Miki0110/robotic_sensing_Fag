import cv2 as cv
from pathlib import Path
from collections import deque

opgave = 4

# importing the videos
vid = ['','','','']
for i in range(3):
    vid[i] = cv.VideoCapture(Path.cwd().parent.as_posix()+'/Lektion_8/materialer/KU_'+str(i+1)+'.avi')

#Implementer en image differencing algoritme
if opgave == 1:
    videoNumber = 1
    #Cap first frame
    frameRef = deque()
    for i in range(3):
        ret, lastFrame = vid[videoNumber].read()
        frameRef.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))

    while(vid[videoNumber].isOpened()):
    # read frame
        ret, frame = vid[videoNumber].read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # subtract with reference frame
        subtracted = cv.absdiff(frameRef.popleft(),gray)

    # update reference frame
        frameRef.append(gray)

    # threshold
        blurVid = cv.medianBlur(subtracted, 5)

        ret, threshVid = cv.threshold(blurVid, 20, 255, cv.THRESH_BINARY)

    # show video
        cv.imshow('frame', threshVid)
    # break if i want to stop the video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid[videoNumber].release()
    cv.destroyAllWindows()

#Prøv med forskellige referencebilleder, f.eks. t-1, t-3 og t-7.
if opgave == 2:
    videoNumber = 2
    # Cap first frame
    frameRef1 = deque()
    frameRef3 = deque()
    frameRef7 = deque()
    for i in range(1):
        ret, lastFrame = vid[videoNumber].read()
        frameRef1.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))
    for i in range(3):
        ret, lastFrame = vid[videoNumber].read()
        frameRef3.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))
    for i in range(7):
        ret, lastFrame = vid[videoNumber].read()
        frameRef7.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))

    while (vid[videoNumber].isOpened()):
        # read frame
        ret, frame = vid[videoNumber].read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # subtract with reference frame
        sub1 = cv.absdiff(frameRef1.popleft(), gray)
        sub3 = cv.absdiff(frameRef3.popleft(), gray)
        sub7 = cv.absdiff(frameRef7.popleft(), gray)

        # update reference frame
        frameRef1.append(gray)
        frameRef3.append(gray)
        frameRef7.append(gray)

        # threshold
        blurVid1 = cv.medianBlur(sub1, 5)
        blurVid3 = cv.medianBlur(sub3, 5)
        blurVid7 = cv.medianBlur(sub7, 5)

        ret, threshVid1 = cv.threshold(blurVid1, 20, 255, cv.THRESH_BINARY)
        ret, threshVid3 = cv.threshold(blurVid3, 20, 255, cv.THRESH_BINARY)
        ret, threshVid7 = cv.threshold(blurVid7, 20, 255, cv.THRESH_BINARY)

        # show video
        cv.imshow('frame1', threshVid1)
        cv.imshow('frame3', threshVid3)
        cv.imshow('frame7', threshVid7)
        # break if i want to stop the video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid[videoNumber].release()
    cv.destroyAllWindows()

#Implementer background subtraction, hvor baggrunden løbende opdateres med denne formel:
# B_new (x, y) = alpha * B_old(x, y) + (1 - alpha)*I(x, y)
if opgave == 3:
    videoNumber = 1
    alpha = 0.9
    # Cap first frame
    frameRef = deque()
    for i in range(3):
        ret, lastFrame = vid[videoNumber].read()
        frameRef.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))

    while (vid[videoNumber].isOpened()):
        # read frame
        ret, frame = vid[videoNumber].read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ref = cv.add(cv.multiply(frameRef.popleft(), alpha), cv.multiply(gray, 1-alpha))
        # subtract with reference frame
        sub = cv.absdiff(ref, gray)

        # update reference frame
        frameRef.append(gray)

        # threshold
        blurVid = cv.medianBlur(sub, 5)

        ret, threshVid = cv.threshold(blurVid, 20, 255, cv.THRESH_BINARY)

        # show video
        cv.imshow('frame', threshVid)
        # break if i want to stop the video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid[videoNumber].release()
    cv.destroyAllWindows()

# Eksperimenter med forskellige alpha-værdier.
    # skal være større end 0.5 for ellers kommer intet i gennem, da jeg trækker det nyeste billede fra det gamle

# Derefter, prøv at bruge livestream fra jeres webkamera
if opgave == 4:
    alpha = 0.9999
    vid = cv.VideoCapture(0)
    # Cap first frame
    frameRef = deque()
    for i in range(20):
        ret, lastFrame = vid.read()
        frameRef.append(cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY))

    while (vid.isOpened()):
        # read frame
        ret, frame = vid.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ref = cv.add(cv.multiply(frameRef.popleft(), alpha), cv.multiply(gray, 1 - alpha))
        # subtract with reference frame
        sub = cv.absdiff(ref, gray)

        # update reference frame
        frameRef.append(gray)

        # threshold
        blurVid = cv.medianBlur(sub, 5)

        ret, threshVid = cv.threshold(blurVid, 20, 255, cv.THRESH_BINARY)

        # show video
        cv.imshow('frame', threshVid)
        # break if i want to stop the video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()