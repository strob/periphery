import numpy
import cv2

def video_frames(src):
    vc = cv2.VideoCapture(src)
    while True:
        retval, im = vc.read()
        if retval:
            yield im #[:,:,[2,1,0]] # BGR->RGB
        else:
            raise StopIteration

def write_video(npv, out):
    vw = cv2.VideoWriter()
    print 'open', vw.open(out, cv2.cv.CV_FOURCC(*'DIVX'), 30.0, (npv.shape[2], npv.shape[1]), True)
    for fr in npv:
        vw.write(fr)


def motionThreshold(npv):
    if len(npv.shape) > 3:
        # greyscale
        npv = npv.mean(axis=3).astype(int)
    else:
        npv = npv.astype(int)
    composite = npv.mean(axis=0).astype(numpy.uint8)
    motion = (npv - composite).clip(0,255).sum(axis=0)
    return motion > (len(npv) * 10)

def largestRegions(thresh, minArea=150, maxRegions=5):
    "return boundingbox of largest contours"
    contours, hierarchy = cv2.findContours(thresh.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = numpy.array([cv2.contourArea(X) for X in contours])

    selection = areas.argsort()[-maxRegions:]
    selection = selection[areas[selection] > minArea]

    print areas[selection]

    return [cv2.boundingRect(X) for X in numpy.array(contours)[selection]]

def multiCinemagraph(src):
    npv = numpy.array([X for X in video_frames(src)])
    mt = motionThreshold(npv)
    import time
    for idx,(x,y,w,h) in enumerate(largestRegions(mt)):
        out= npv.copy()
        for f_idx,fr in enumerate(npv):
            out[f_idx] = npv[0]
            out[f_idx,y:y+h,x:x+w] = fr[y:y+h,x:x+w]
        write_video(out, '%s.%d.avi' % (src, idx))

if __name__=='__main__':
    import sys
    for video in sys.argv[1:]:
        multiCinemagraph(video)
