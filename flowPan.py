# coding=utf-8

import numpy
import cv2
import os

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
    # pseudo-cache
    if os.path.exists(src + '.affinity.1.npy'):
        return numpy.load(src + '.affinity.1.npy')

    print 'cache fail'

    cap = cv2.VideoCapture(src)
    affs = []
    prev = None
    while True:
        retval, im = cap.read()
        if retval:
            if prev is not None:
                aff = cv2.estimateRigidTransform(im, prev, True)
                affs.append(aff)
            prev = im
        else:
            affs = numpy.array(affs)
            numpy.save(src + '.affinity.1.npy', affs)
            return affs

def affine(im1, im2):
    return cv2.estimateRigidTransform(im1, im2, True)

def previewStabilization(src):
    cv2.namedWindow('frame')

    cap = cv2.VideoCapture(src)
    rv, im = cap.read()
    prev = None
    accaff = None
    while rv:
        if prev is not None:
            aff = affine(im, prev)
            if accaff is None:
                cv2.imshow('frame', im)
                accaff = aff
            else:
                accaff = compose(aff, accaff)
                cv2.imshow('frame', cv2.warpAffine(im, accaff, (im.shape[1], im.shape[0])))

        prev = im
        rv, im = cap.read()

        if cv2.waitKey(50) == 27: # escape
            break

def spatialize(src):
    affs = affinity(src)
    print len(affs), 'affine frames'

    # For now, naively accumulate the transformation.
    # It may be worth wiping the non-translation component every so often.

    accaffs = [affs[0]]
    for a in affs[1:]:
        accaffs.append(compose(a, accaffs[-1]))

    return numpy.array(accaffs)

def track(affs, pt=[0,0]):
    "return a pointlist tracking the provided"
    pts = []
    for A in affs:
        A = numpy.array(A)
        A = numpy.matrix([A[0], A[1], [0, 0, 1]])
        pts.append(numpy.array(A * numpy.matrix([pt[0], pt[1], 1]).T).reshape(-1)[:2])
    return numpy.array(pts)

def drift_guard(src, maxt=5000, maxdistort=10, W=1280, H=720):
    affs = spatialize(src)

    print len(affs), 'frames'

    # Check when we lose it.

    # too translated
    too_far = abs(affs[:,:,2]).reshape((-1,2)) > maxt
    far_idx = max(too_far[:,0].argmax(), too_far[:,1].argmax())
    if far_idx > 0:
        print 'too far at frame', far_idx, ':', affs[far_idx]
        affs = affs[:far_idx]

    # too sheared
    topleft = track(affs, pt=[0,0])
    topright = track(affs, pt=[W,0])
    bottomright = track(affs, pt=[W,H])
    bottomleft = track(affs, pt=[0,H])

    diag1 = numpy.hypot(*abs(bottomright - topleft).T)
    diag2 = numpy.hypot(*abs(topright - bottomleft).T)

    ratio = maxdistort * 999 * numpy.ones(len(diag1))
    ratio[diag2 != 0] = diag1[diag2 != 0] / diag2[diag2 != 0]
    ratio[ratio < 1] = 1.0 / ratio[ratio < 1]

    print 'min ratio', ratio.min(), 'max', ratio.max(), 'mean', ratio.mean()
    
    too_sheared_idx = (ratio > maxdistort).argmax()
    if too_sheared_idx > 0:
        print 'too sheared at frame', too_sheared_idx, ':', affs[too_sheared_idx]
        affs = affs[:too_sheared_idx]

    return affs

def fit(affs, in_size, out_size):
    "return a set of affine transformations that scale and translate into the output size"

    W = in_size[0]
    H = in_size[1]

    # Compute a bounding box.
    topleft = track(affs, pt=[0,0])
    topright = track(affs, pt=[W,0])
    bottomright = track(affs, pt=[W,H])
    bottomleft = track(affs, pt=[0,H])

    x0 = numpy.concatenate([topleft, bottomleft])[:,0].min()
    x1 = numpy.concatenate([topright, bottomright])[:,0].max()
    y0 = numpy.concatenate([topleft, topright])[:,1].min()
    y1 = numpy.concatenate([bottomleft, bottomright])[:,1].max()

    tx = -x0
    ty = -y0

    translateA = [[1,0,tx], [0,1,ty]]

    scale = min(float(out_size[0])/(x1-x0), 
                float(out_size[1])/(y1-y0))

    scaleA = [[scale, 0, 0], [0, scale, 0]]

    print translateA, scaleA

    #transform = [[scale, 0, tx], [0, scale, ty]]
    transform = compose(scaleA, translateA)
    # transform = [[1,0,0],[0,1,0]]

    return numpy.array([compose(transform, X) for X in affs])

def previewFitted(src):
    cv2.namedWindow('frame')

    cap = cv2.VideoCapture(src)
    rv, im = cap.read()
    idx = 0

    #transforms = fit(drift_guard(src), (im.shape[1], im.shape[0]), (1366,750))
    transforms = fit(drift_guard(src), (im.shape[1], im.shape[0]), (im.shape[1], im.shape[0]))
    out = numpy.zeros((750, 1366,3), dtype=numpy.uint8)

    while rv and idx < len(transforms)-1:
        if idx > 0:
            aff = transforms[idx-1]
            # print aff
            warp = cv2.warpAffine(im, aff, (out.shape[1], out.shape[0]))
            # XXX: blend? keep track of corners more carefully?
            out[warp>0] = warp[warp>0]
            cv2.imshow('frame', out)

        idx += 1
        rv, im = cap.read()

        if cv2.waitKey(50) == 27: # escape
            break

    if cv2.waitKey(-1) == 27: # escape
        pass


if __name__=='__main__':
    import sys
    for src in sys.argv[1:]:
        #aff = affinity(src)
        # previewStabilization(src)
        previewFitted(src)
