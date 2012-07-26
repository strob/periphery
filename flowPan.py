# coding=utf-8

import numpy
import cv2

# cv2.estimateRigidTransform(src, dst, fullAffine) → retval

# Assemble video frames into a moving panorama

# accumulate the affine transformation; when the translation is
# greater than a certain amount, stop. basta.

def compose(f, g):
    "f(g(x)) = (f * g) * x"

    f = numpy.array(f)
    g = numpy.array(g)

    f0 = numpy.matrix([f[0], f[1], [0, 0, 1]])
    g0 = numpy.matrix([g[0], g[1], [0, 0, 1]])
    return numpy.array(f0 * g0)[:2]

def affinity(src):
    cap = cv2.VideoCapture(src)
    affs = []
    prev = None
    while True:
        retval, im = cap.read()
        if retval:
            if prev is not None:
                aff = cv2.estimateRigidTransform(prev, im, True)
                affs.append(aff)
            prev = im
        else:
            return numpy.array(affs)

def affine(im1, im2):
    return cv2.estimateRigidTransform(im1, im2, True)

if __name__=='__main__':
    import sys
    src = sys.argv[1]
    # aff = affinity(src)

    TRAIL = 30
    CBUFFER = []

    cv2.namedWindow('frame')

    cap = cv2.VideoCapture(src)
    rv, im = cap.read()

    while rv:
        if len(CBUFFER) == TRAIL:
            accaff = None
            for i in range(len(CBUFFER)-1):
                a = CBUFFER[i]
                b = CBUFFER[i+1]
                aff = affine(b,a)
                if accaff is None:
                    cv2.imshow('frame', a)
                    accaff = aff
                else:
                    accaff = compose(aff, accaff)

                if cv2.waitKey(20) == 27: # escape
                    break
                cv2.imshow('frame', cv2.warpAffine(b, accaff, (a.shape[1], a.shape[0])))

        rv, im = cap.read()
        CBUFFER.append(im)
        if len(CBUFFER) > TRAIL:
            CBUFFER.pop(0)

        if cv2.waitKey(50) == 27: # escape
            break
