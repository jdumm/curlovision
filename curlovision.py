from __future__ import print_function
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import math
from PIL import Image
import pytesseract
import re
from collections import deque
from copy import deepcopy
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection

# TODO: Abstract out all match-specific parameters.

# Processes an entire video, returns a MatchResult.
def process_video(vid_filename, frame_start=0, nskip=5, icon_filenames=['stone_icons/red_icon_Olympics2018.npy','stone_icons/yel_icon_Olympics2018.npy'],draw_key_frames=False, draw_test_frames=False, debug_12ft=False, debug_stones=False, debug_scoreboard=False):
    # Load saved image templates for 'stones remaining' icons:
    # TODO: Check load success
    red_icon = np.load(icon_filenames[0])
    yel_icon = np.load(icon_filenames[1])

    cap = cv2.VideoCapture(vid_filename)

    fcount=frame_start
    cap.set(1,frame_start)
    nskip = 5
    match_result = MatchResult(vid_filename)
    stm = ShortTermMemory(15)
    stm.verbose = False

    stm.frame_cache  = deque(maxlen=15)
    stm.layout_cache = deque(maxlen=15)
    stm.sbs_cache    = deque(maxlen=15)

    vid_end_found = False
    while(cap.isOpened()):
        for i in range(nskip):
            ret, frame = cap.read()
            #if not ret: break # Corrupt frame or end of file
            if not ret:
                vid_end_found = True
                break # Corrupt frame or end of file
            fcount += 1

        if vid_end_found: 
            # Package last EndResult and return the MatchResult
            match_result.add_end_result(deepcopy(stm.current_end_result))  # Store a copy
            return match_result
    
        # For monitoring where we are in the video:
        if draw_test_frames and fcount%900 == 0: # print test frame every so often (900=1min @ 15fps)
            print("Test frame: {}".format(fcount))
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()

        p2,minR,maxR = infer12ft(frame,draw=debug_12ft)
        if (p2,minR,maxR) != (0,0,0): # Valid 12ft inferred.  Begin collecting frame sequence.
            stm.frame_cache.append(frame)
        elif len(stm.frame_cache) >= 3: # threshold of 3 indicates persistent overhead camera view of the house
            #print('Processing frame_cache of size {}'.format(len(stm.frame_cache)))
            for frame in stm.frame_cache:
                layout,dframe,sbstate = process_frame(frame,icons=(red_icon,yel_icon),draw=False,debug_stones=debug_stones,debug_scoreboard=debug_scoreboard)
                if layout is None or sbstate is None or sbstate.current_end < stm.end_prev: # skip invalid frames and replays
                    continue

                # First check if the end number has changed, in which case return the EndResult.
                if(sbstate.current_end>0 and sbstate.current_end<=20 and # Just some sanity checks
                   sbstate.current_end > stm.end_prev): # A new end! Package and return the last one.

                    stm.current_end_result.end_num = stm.end_prev
                    stm.current_end_result.red_score = sbstate.current_score_red
                    stm.current_end_result.yel_score = sbstate.current_score_yel

                    stm.n_red_prev = 8
                    stm.n_yel_prev = 8
                    stm.end_prev = sbstate.current_end
                    stm.layout_cache.clear()
                    stm.sbstate_cache.clear()
                    if stm.verbose: print(stm.current_end_result)
                    match_result.add_end_result(deepcopy(stm.current_end_result))  # Store a copy
                    stm.current_end_result = EndResult() # re-initialize

                stm.layout_cache.append(layout)
                stm.dframe_cache.append(dframe)
                stm.sbstate_cache.append(sbstate)

            # Now that the cache is processed, find the best layout, hammer, and stones left
            ibest = stm.find_best_layout()
            if ibest>=0: # Valid best-frame of cache found:
                key_frame = fcount-(len(stm.frame_cache)-ibest)*nskip # TODO: Cache key frames for fast reprocessing
                if draw_key_frames:
                    print('key frame: {}'.format(key_frame))
                    plt.figure(figsize=(15,15))
                    plt.imshow( stm.dframe_cache[ibest] )
                    plt.show()
                n_red,n_yel = stm.find_nstones_stats()
                #if stm.verbose: print("{} {} {} {}".format(n_red,n_yel,stm.n_red_prev,stm.n_yel_prev))
                if stm.current_end_result.red_hammer == -1: # If it's unset yet, check if we have enough info to do so now
                    if   n_yel < n_red: stm.current_end_result.red_hammer = 1
                    elif n_yel > n_red: stm.current_end_result.red_hammer = 0

                # If stones left are the same as before, we can replace the previous layout.
                #  E.g. the camera panned away, perhaps during a timeout, then came back to the same top-down view.
                if n_red >= stm.n_red_prev and n_yel >= stm.n_yel_prev:
                    #if stm.verbose: print('Duplicate frame for same stone, updating layout...')
                    # Replace last layout:
                    stm.current_end_result.stone_layouts[-1] = stm.layout_cache[ibest]
                else:
                    stm.current_end_result.add_stone_layout(stm.layout_cache[ibest])
                    stm.current_end_result.add_stones_left(n_red,n_yel)
                    stm.n_red_prev,stm.n_yel_prev = n_red,n_yel
                    # Running score and end num (mostly for the 10th end)
                    stm.current_end_result.end_num = stm.sbstate_cache[ibest].current_end
                    stm.current_end_result.red_score = stm.sbstate_cache[ibest].current_score_red
                    stm.current_end_result.yel_score = stm.sbstate_cache[ibest].current_score_yel

            # Clear caches to prepare for the next stone
            stm.clear_caches()

# Defunct stone color finder
def stone_color_finder_rgb(cframe,stone,draw=False):
    x,y,r = stone[0], stone[1], stone[2]
    r = int(r*0.5)
    roi = cframe[y-r:y+r+1,x-r-1:x+r+1]
    if draw:
        plt.imshow(roi)
        plt.show()
        print(np.median(roi,axis=(0,1)))
    #rgb = np.mean(roi,axis=(0,1))
    rgb = np.median(roi,axis=(0,1))
    #rgb = [ 137.08605974,   54.62802276,   54.01493599]
    if rgb[2]>65: return None # invalid
    elif (rgb[1]>122 and rgb[0]>133): return 'yellow' # Yellow
    elif (rgb[0]>130 and rgb[1]<20 and rgb[2]<20): return None # Bright red in non-dim image - false positive
    elif (rgb[0]>113 and rgb[1]>10 and rgb[2]>9): return 'red' # Red
    else: return None # invalid

# Defunct stone color finder
def stone_color_finder_hsv(cframe,stone,draw=False): # Defunct
    x,y,r = stone[0], stone[1], stone[2]
    h,w,_ = cframe.shape
    stone_frac = 0.55 # Zoom in factor based on radius of stone
    r = max(int(r*stone_frac),6)
    roi = cframe[max(y-r-9,0):min(y+r+2,h),max(x-r-5,0):min(x+r+5,w)]
    #roi = cframe[max(y-r,0):min(y+r,h),max(x-r,0):min(x+r,w)]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    circ = []  # a list of pixels in a circular radius from center
    h,w,_ = roi_hsv.shape
    for y in range(h):
        for x in range(w):
            dist = math.sqrt((w/2-x)**2+(h/2-y)**2)
            if dist < (w+h+3)/4: circ.append(roi_hsv[y,x])
    hsv = np.median(np.array(circ),axis=0)
    #hsv = np.median(roi_hsv,axis=(0,1))
    if draw:
        plt.imshow(roi)
        plt.show()
        print(hsv)
        print(np.median(roi_hsv,axis=(0,1)))
    redh_low,redh_hi = 155,181  # 175
    reds_low,reds_hi = 90 ,240  # 190
    redv_low,redv_hi = 90 ,150  # 123
    yelh_low,yelh_hi = 18 ,30   # 27
    yels_low,yels_hi = 90 ,240  # 175
    yelv_low,yelv_hi = 90 ,200  # 180
    color = None
    # Find primary (median) color:
    if ( hsv[0]>=redh_low and hsv[0]<=redh_hi and
         hsv[1]>=reds_low and hsv[1]<=reds_hi and
         hsv[2]>=redv_low and hsv[2]<=redv_hi):  color = 'red'
    elif(hsv[0]>=yelh_low and hsv[0]<=yelh_hi and
         hsv[1]>=yels_low and hsv[1]<=yels_hi and
         hsv[2]>=yelv_low and hsv[2]<=yelv_hi):  color = 'yellow'
    else: color = None
    # Check for consistency (is the roi primarily just this one color?)
    mframe = cv2.inRange(roi_hsv,hsv-[15,100,70],hsv+[15,60,60])
    h,w,_ = roi.shape
    mfrac = len(mframe[mframe>0])/float(w*h)
    if draw: print(mfrac, color)
    if (mfrac > 0.10 and mfrac < 0.95): return color  # The image should be mostly the same color but not fully
    return None

# Stone color finder.  Rejects false positives that are not sufficiently red or yellow.
def stone_color_matcher(cframe,stone,draw=False):
    x,y,r = stone[0], stone[1], stone[2]
    h,w,_ = cframe.shape
    stone_frac = 0.55 # Zoom in factor based on radius of stone
    r = max(int(r*stone_frac),6)
    roi = cframe[max(y-r-7,0):min(y+r+4,h),max(x-r-5,0):min(x+r+5,w)]
    #roi = cframe[max(y-r,0):min(y+r,h),max(x-r,0):min(x+r,w)]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h,w,_ = roi_hsv.shape
    if draw:
        plt.imshow(roi)
        plt.show()
    # Check for fraction of stone that's red, yellow, or neither
    roi_red = cv2.inRange(roi_hsv,np.array([155,90,90]),np.array([181,240,150]))
    roi_yel = cv2.inRange(roi_hsv,np.array([18,90,90]),np.array([30,240,200]))
    frac_red = len(roi_red[roi_red>0])/float(h*w)
    frac_yel = len(roi_yel[roi_yel>0])/float(h*w)
    color = None
    if frac_red>0.23 and frac_red<0.90: color = 'red'
    if frac_yel>0.23 and frac_yel<0.90: color = 'yellow'
    if draw: print('Red Frac: {:.3f}, Yel Frac: {:.3f}:   Color: {}'.format(frac_red, frac_yel,color))
    return color

# Check BGR -> RGB frame for how much is blue.  First step in deciding to process the frame further
#  so it needs to be super fast.
def infer12ft(frame,draw=False):
    cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blue12 = [32.,  115.,  115.]  # From Olympic Ice, determined by hand

    cw = 30 # Seach window size around the true blue12 color in each rgb channel
    bl = (blue12[0]-cw,blue12[1]-cw,blue12[2]-cw)
    bu = (blue12[0]+cw,blue12[1]+cw,blue12[2]+cw)

    h,w,_ = cframe.shape
    bframe = cv2.inRange(cframe,bl,bu)
    #plt.imshow(bframe)
    bframe = bframe[:,int(w/5.):-int(w/5.)]
    bfrac = len(bframe[bframe>0])/float(w*h)
    if draw:
        plt.imshow(bframe,cmap='gray')
        plt.show()
        print("Valid Blue Fraction Found: {}".format(bfrac))
    if bfrac < 0.06:  # Pretty low threshold.  False positives will be sorted out later.
        #frame_status.blue_found = False
        return 0,0,0
    #else: frame_status.blue_found = True
    # Check for cutoff 12ft on bottom of frame:
    botsize = 10 # in pixels
    botbframe = bframe[-botsize:]
    bbot  = len(botbframe[botbframe>0])/float(w*botsize)
    if bbot>0.12: bfrac *= 1+bbot # a little adjustment for cutoff 12ft
    #print(bfrac)
    p2,minR,maxR = 70,280,440
    # These determined from trial and error:
    if bfrac< 0.09               : p2,minR,maxR=65,280,410
    if bfrac>=0.09 and bfrac<0.12: p2,minR,maxR=60,340,440
    if bfrac>=0.12 and bfrac<0.16: p2,minR,maxR=58,360,460
    if bfrac>=0.16               : p2,minR,maxR=55,420,520
    
    #print(p2,minR,maxR)
    return p2,minR,maxR

# Used to identify 12ft ring in the house
def find12ft(gframe,p2,minR,maxR,zy=0.0,zw=0.0,rec_count=0):
    rec_count += 1
    # Termination criteria, failure to find any rings:
    #print(p2,minR,maxR)
    if p2<50 or p2>100 or rec_count>5: return 0,0,0
    
    # Initial pass:
    h,w = gframe.shape
    zframe = gframe[int(h*zy):,int(w*zw):int(-w*zw-1)] # Zoom in a bit
    twelves = cv2.HoughCircles(zframe,cv2.HOUGH_GRADIENT,1,20,
                               param1=200,param2=p2,minRadius=minR,maxRadius=maxR)
    #print("nTwelves: " ,len(twelves[0,:]))
    #print("Twelves: " ,twelves[0,:])
    if twelves is None: # lower noise threshold, change ring size, change zoom
        return find12ft(gframe,p2-5,minR-20,maxR+20)    
    elif len(twelves[0,:])>5: # tighten criteria
        return find12ft(gframe,p2+2,minR+2,maxR-1)
    else: # Find best 12-ft ring
        x,y,r = 0,0,0
        twelves = np.uint16(np.around(twelves))
        twelves = twelves[0,:]
        xs, ys, rs = [], [], []
        for ring in twelves:
            #print(ring)
            xs.append(ring[0]), ys.append(ring[1]), rs.append(ring[2])
            #cv2.circle(dframe,(ring[0],ring[1]),ring[2],(0,225,0),2)
        if len(twelves[0,:])==2: x,y,r = int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs))
        else:                    x,y,r = int(np.median(xs)), int(np.median(ys)), int(np.median(rs))
    return x,y,r

# Used to identify all rings in the house besides the 12ft
def find_smaller_rings(gframe,r_in,p2=50,rwin=10,rec_count=0):
    r_tol = 0.01
    rec_count += 1
    minR,maxR = int(r_in*(1-r_tol)-rwin),int(r_in*(1+r_tol)+rwin) 
    # Termination criteria, failure to find any rings:
    #print(p2,minR,maxR)
    if p2<25 or p2>70 or rec_count>10: return 0,0,0
    #h,w = gframe.shape
    #zframe = gframe[int(h*zy):,int(w*zw):int(-w*zw-1)] # Zoom in a bit from the top and sides

    rings = cv2.HoughCircles(gframe,cv2.HOUGH_GRADIENT,1,25,
                             param1=200,param2=p2,minRadius=minR,maxRadius=maxR)
    if rings is None: # Lower noise, increase ring size
        return find_smaller_rings(gframe,r_in,p2-2,rwin+2,rec_count)
    elif len(rings[0,:])>3: # tighten criteria
        return find_smaller_rings(gframe,r_in,p2+2,rwin+1,rec_count)
    else: # Find best 4-ft ring
        x,y,r = 0,0,0
        rings = np.uint16(np.around(rings))
        rings = rings[0,:]
        xs, ys, rs = [], [], []
        for ring in rings:
            #print(ring)
            xs.append(ring[0]), ys.append(ring[1]), rs.append(ring[2])
            #cv2.circle(dframe,(ring[0],ring[1]),ring[2],(0,225,0),2)
        if len(rings[0,:])==2: x,y,r = int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs))
        else:                  x,y,r = int(np.median(xs)), int(np.median(ys)), int(np.median(rs))
    return x,y,r

# Used to identify all stones in the frame with a Hough Transform, rejecting false positives that are not red or yellow
def find_stones(gframe,cframe,r_in,p2=50,rwin=5,debug=False):
    r_tol = 0.005
    minR,maxR = int(r_in*(1-r_tol)-rwin-7),int(r_in*(1+r_tol)+rwin-1) 
    #minR,maxR = int(r_in*(1-r_tol)-rwin-10),int(r_in*(1+r_tol)+rwin+5) 
    stones  = cv2.HoughCircles(gframe,cv2.HOUGH_GRADIENT,1,22,
                               param1=50,param2=27,minRadius=minR,maxRadius=maxR)
    stones_ret = np.empty((0,4)) # X,Y,R,color(str)
    if stones is not None:
        stones = np.uint16(np.around(stones))
        stones = stones[0,:]
        # Color rejection:
        for i,stone in enumerate(stones):
            #color = stone_color_finder_rgb(cframe,stone,False)
            #color = stone_color_finder_hsv(cframe,stone,debug)
            color = stone_color_matcher(cframe,stone,debug)
            if color is None: continue # Invalid stone color: not red or yellow so false positive.
            if color is 'red': stone_rgb = (255,0,0)
            elif color is 'yellow': stone_rgb = (255,255,0)
            else: continue         
            stones_ret = np.append(stones_ret,[[stone[0],stone[1],stone[2],color]],axis=0)
            #print('\t\t{} \t{:.2f}\t{:.2f}'.format(color,(stone[0]-x)*p2f ,-p2f*(stone[1]-y)))
    if debug:
        print("Rin,MinR,MaxR: {}, {}, {}".format(r_in,minR,maxR))
        print(stones_ret)

    return stones_ret

# Uses template matching to count how many stones are remaining to be thrown in the end.  Needs pre-loaded icon templates.
def read_stones_remaining_icons(cframe,dframe,icons,draw=False):
    if len(icons) != 2: print('ERROR: Expected a list of red and yellow icons')
    red_icon = icons[0]
    #h,w,_ = red_icon.shape
    #red_icon = red_icon[int(h*0.10):int(h*0.90),:]
    yel_icon = icons[1]
    #h,w,_ = yel_icon.shape
    #yel_icon = yel_icon[int(h*0.10):int(h*0.90),:]
    h,w,_ = cframe.shape
    red_roi =  cframe[int(0.05*h):int(0.10*h),int(0.22*w):int(0.49*w)]  # Just examine upper left corner
    red_droi = dframe[int(0.05*h):int(0.10*h),int(0.22*w):int(0.49*w)]  # Same region for draw frame
    yel_roi =  cframe[int(0.10*h):int(0.15*h),int(0.22*w):int(0.49*w)]  # Just examine upper left corner
    yel_droi = dframe[int(0.10*h):int(0.15*h),int(0.22*w):int(0.49*w)]  # Same region for draw frame
    red_res = cv2.matchTemplate(red_roi,red_icon,cv2.TM_CCOEFF_NORMED)
    red_thresh = np.max(red_res)-0.03 # Dynamic threshold, within 1.5% of the peak template match result
    yel_res = cv2.matchTemplate(yel_roi,yel_icon,cv2.TM_CCOEFF_NORMED)
    yel_thresh = np.max(yel_res)-0.03 # Dynamic threshold, within 1.5% of the peak template match result
    #print(red_thresh,yel_thresh)
    if red_thresh < 0.80 and yel_thresh < 0.80: return -1,-1  # No icons found... weird so return invalid
    h,w,_ = red_icon.shape
    red_loc = np.where( red_res >= red_thresh)
    yel_loc = np.where( yel_res >= yel_thresh)
    # Ensure no overlapping matches that are too close, left to right (This avoids double counting)
    red_locs = np.sort(np.array(red_loc[1]))
    red_difs = np.ediff1d(red_locs, to_begin=999)
    red_difs = np.where(red_difs>20)  # Require 20 pix separation
    if red_thresh < 0.80: nred=0
    else: nred = len(zip(*red_difs[::-1]))
    yel_locs = np.sort(np.array(yel_loc[1]))
    yel_difs = np.ediff1d(yel_locs, to_begin=999)
    yel_difs = np.where(yel_difs>20)  # Require 20 pix separation
    if yel_thresh < 0.80: nyel=0
    else: nyel = len(zip(*yel_difs[::-1]))
    #print(yel_res[red_loc])
    #print(yel_res[yel_loc])
    # Draw squares over the icons:
    if nred>0:
        for pt in zip(*red_loc[::-1]):
            cv2.rectangle(red_droi, (pt[0]-5,pt[1]-10), (pt[0]+w+5, pt[1]+h+10), (255,0,0), 2)
    if nyel>0:
        for pt in zip(*yel_loc[::-1]):
            cv2.rectangle(yel_droi, (pt[0]-5,pt[1]-10), (pt[0]+w+5, pt[1]+h+10), (255,255,0), 2)
    if draw==True:
        plt.imshow(red_roi)
        plt.show()
        plt.imshow(red_res)
        plt.show()
        plt.imshow(yel_roi)
        plt.show()
        plt.imshow(yel_res)
        plt.show()
    return nred,nyel

# Uses pytesseract to read the digits of the current end
def read_current_end_text(cframe,dframe,draw=False):
    y1,y2,x1,x2 = 168,212,295,345 # Custom boundaries for Olympics2018
    roi = cframe[y1:y2,x1:x2]
    cv2.rectangle(dframe, (x1,y1), (x2,y2), (155,155,155), 2)
    groi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(Image.fromarray(groi),config='-psm 8 -c tessedit_char_whitelist=0123456789')
    #text = pytesseract.image_to_string(Image.fromarray(groi),config='outputbase digits')
    try:
        current_end = re.search('\d+',text).group(0)
    except:
        current_end = -1
    font = cv2.FONT_HERSHEY_DUPLEX
    h,w,_ = roi.shape
    cv2.putText(dframe,str(current_end),(297,245), font, 1.4,(255,255,0),2,cv2.LINE_AA)
    if draw:
        print(text,current_end)
        plt.figure()
        plt.imshow(roi)
        plt.show()
    return current_end

# Uses pytesseract to read the digits of the current score
def read_current_score_text(cframe,dframe,draw=False):
    y1,y2,x1,x2 = 56,96,362,418 # Custom boundaries for Olympics2018
    roi1 = cframe[y1:y2,x1:x2]
    cv2.rectangle(dframe, (x1,y1), (x2,y2), (155,155,155), 2)
    y1,y2,x1,x2 = 113,153,363,419 # Custom boundaries for Olympics2018
    roi2 = cframe[y1:y2,x1:x2]
    cv2.rectangle(dframe, (x1,y1), (x2,y2), (155,155,155), 2)

    groi1= cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    groi2= cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    text1 = pytesseract.image_to_string(Image.fromarray(groi1),config='-psm 8 -c tessedit_char_whitelist=0123456789')
    text2 = pytesseract.image_to_string(Image.fromarray(groi2),config='-psm 8 -c tessedit_char_whitelist=0123456789')
    #text1 = pytesseract.image_to_string(Image.fromarray(groi1),config='-psm 8')
    #text2 = pytesseract.image_to_string(Image.fromarray(groi2),config='-psm 8')
    try:
        current_score_red = re.search('\d+',text1).group(0)
    except:
        current_score_red = -1
    try:
        current_score_yel = re.search('\d+',text2).group(0)
    except:
        current_score_yel = -1
    font = cv2.FONT_HERSHEY_DUPLEX
    h,w,_ = roi1.shape
    cv2.putText(dframe,str(current_score_red),(370,44 ), font, 1.6,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(dframe,str(current_score_yel),(370,193), font, 1.6,(255,255,0),2,cv2.LINE_AA)
    if draw==True:
        print(text1,current_score_red)
        plt.figure()
        plt.imshow(roi1)
        plt.show()
        print(text2,current_score_yel)
        plt.figure()
        plt.imshow(roi2)
        plt.show()
    return current_score_red,current_score_yel

# Reads the current end, scoreboard, and determines the number of stones left to be thrown in the end
def analyze_game_status(cframe,dframe,icons,draw=True):

    current_end =                         read_current_end_text       (cframe,dframe,draw=draw)
    current_score_red,current_score_yel = read_current_score_text     (cframe,dframe,draw=draw)
    nred,nyel =                           read_stones_remaining_icons (cframe,dframe,icons,draw=draw)

    return ScoreboardState(current_end,current_score_red,current_score_yel,nred,nyel)


# High-level flow for how to process important frames
def process_frame(frame,icons=None,draw=False,saveas=None,debug_stones=False,debug_scoreboard=False):
    cframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dframe = cframe.copy()
    gframe = cv2.medianBlur(frame,5)
    gframe = cv2.cvtColor(gframe, cv2.COLOR_BGR2GRAY)
    h,w = gframe.shape
    invalid_ret = None,None,None
    #invalid_ret = None,None,ScoreboardState()

    # Infer the 12-ft using color check
    p2,minR,maxR = infer12ft(frame,draw=False)
    if ((p2,minR,maxR)==(0,0,0)):
        #print("Invalid Frame!")
        return invalid_ret

    # Find best 12-ft using Hough Transforms
    x12,y12,r12 = find12ft(gframe,p2,minR,maxR)
    if (x12,y12,r12)==(0,0,0) or r12<200 or x12<r12 or x12>w-r12 or y12<r12: # Sometimes a wild circle appears near the edge
        #print("Invalid Frame!")
        return invalid_ret
    #print(r12)
    # draw 12 ft ring and center:
    cv2.circle(dframe,(x12,y12),r12,(0,255,255),10)
    cv2.circle(dframe,(x12,y12),2,(0,255,255),15)    

    # The 8 ft should be 2/3 as big as the 12 ft
    #print(y12,x12,r12)
    #plt.imshow(dframe)
    #plt.show()
    zframe = gframe[y12-r12:y12+r12,x12-r12:x12+r12]
    x8,y8,r8 = find_smaller_rings(zframe,r12*2./3.,p2=60)
    if (x8,y8,r8)==(0,0,0) or r8<100 or x8<r8 or x8>w-r8 or y8<r8:
        #print("Invalid Frame!")
        return invalid_ret

    # Add back offsets from zooming:
    x8,y8 = x8+x12-r12,y8+y12-r12
    # draw 8 ft ring and center:
    cv2.circle(dframe,(x8,y8),r8,(255,255,255),10)
    cv2.circle(dframe,(x8,y8),2,(255,255,255),10)    

    # The 4 ft should be 1/2 as big as the 8 ft
    zframe = gframe[y8-r8:y8+r8,x8-r8:x8+r8]
    x4,y4,r4 = find_smaller_rings(zframe,r8*1./2.,p2=60)
    if (x4,y4,r4)==(0,0,0):
        #print("Invalid Frame!")
        return invalid_ret
    x4,y4 = x4+x8-r8,y4+y8-r8
    #print(r4)
    # draw 4 ft ring and center:
    cv2.circle(dframe,(x4,y4),r4,(255,0,0),10)
    cv2.circle(dframe,(x4,y4),2,(255,0,0),5)    

    # The button is a little less than 1/2 as big as the 4 ft in Korea
    zframe = gframe[y4-r4:y4+r4,x4-r4:x4+r4]
    x2,y2,r2 = find_smaller_rings(zframe,r4*1./2.,p2=40)
    if (x2,y2,r2)==(0,0,0):
        #print("Invalid Frame!")
        return invalid_ret
    x2,y2 = x2+x4-r4,y2+y4-r4
    #print(r2)
    # draw 2 ft ring and center:
    cv2.circle(dframe,(x2,y2),r2,(255,255,255),10)
    cv2.circle(dframe,(x2,y2),2,(255,255,255),2)    

    # Find pixel-to-ft conversion factor:
    #p2f = fudge_factor*4./(r4*2) # pixels to feet conversion using the 4ft measurement
    #print(r2/float(r4))
    # The button appears to be 1 ft 4 inches in Korea or 1.33 ft!
    fudge_factor = 1.5
    p2f = fudge_factor*1./(r2*2) # pixels to feet conversion using the button measurement
    #print(r2,p2f)
    # Reconstruct the 12ft based on this measurement:
    cv2.circle(dframe,(x2,y2),int(6/p2f),(255,255,255),3)
    
    # TODO: Analyze reliability of frame by comparing 4ft and button:
    #diff24 = math.sqrt(math.pow(x4-x2,2)+math.pow(y4-y2,2))
    #ratio24 = r2/float(r4)
    #print("{:.2f}".format(diff24))
    
    # A stone's max circumference is 36 in.  36/pi ~ 11.5 in max diameter or a bit less than 1/4 of the 4 ft 
    #stones = find_stones(gframe,cframe,r4*1./4.,p2=40,debug=True)
    stones = find_stones(gframe,cframe,int(0.9583/2./p2f),p2=40,debug=debug_stones)
    #print(stones)

    stones_ret = []
    for stone in stones:
        #print(stone)
        if stone[3] == 'red': stone_rgb = (255,0,0)
        elif stone[3] == 'yellow': stone_rgb = (255,255,0)
        else: print('ERROR!  Invalid stone color {} should have been checked already'.format(stone[3]))
        cv2.circle(dframe,(int(stone[0]),int(stone[1])),int(stone[2]),stone_rgb,3)
        # draw the center of the circle
        cv2.circle(dframe,(int(stone[0]),int(stone[1])),3,stone_rgb,5)
        #print('\t\t\t{}\t{}'.format(stone[0] ,stone[1]))

        # Convert stone positions from pixels to ft and filter stones outside a reasonable boundary:
        x =  p2f*(int(stone[0])-x2)
        y = -p2f*(int(stone[1])-y2)
        if x<-7 or x>7 or y<-8 or y>20: continue   # Hardcoded acceptable bounds of the field of play
        stones_ret.append(Stone(p2f*(int(stone[0])-x2), -p2f*(int(stone[1])-y2), p2f*int(stone[2]), stone[3]))
        if draw==True: print('\t\t{} \t{:.2f}\t{:.2f}'.format(stone[3],float(stone[0])-x2,float(stone[1])-y2))

    # Check game status (number of stones remaining for each team):
    n_red_left,n_yel_left = -1,-1
    if icons is not None:
        #current_end,current_score_red,current_score_yel,n_red_left,n_yel_left = analyze_game_status(cframe,dframe,icons)
        sbs = analyze_game_status(cframe,dframe,icons,draw=debug_scoreboard)
        if not sbs.valid(): return invalid_ret
        if draw==True:
            print('Current End: {}'.format(sbs.current_end))
            print('Current Score: {} to {}'.format(sbs.current_score_red,sbs.current_score_yel))
            print('Remaining Red Stones: {}, Yellow Stones: {}'.format(sbs.n_red_left,sbs.n_yel_left))

    if draw==True:
        plt.figure(figsize=(15,15))
        #plt.imshow(gframe,cmap='gray')
        plt.imshow(dframe)
        if saveas is not None: plt.savefig(saveas)
        plt.show()
    return StoneLayout(stones_ret),dframe,sbs


# Basic stone data: position, size, and color
class Stone:
    def __init__(self,x=0,y=0,r=0,color=None):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        self.color = str(color)
    def __str__(self):
        return "\t\t\t\t{:7}\t\t{:.2f}\t\t{:.2f}".format(self.color, self.x, self.y)

# Positions of stones and game status.  Can calculate score of iteslf and draw itself.
class StoneLayout:
    def __init__(self,stones=[]):
        self.stone_radius = 11.5/12./2. # Fixed number in feet, based on 36-inch circumference in rules
        self.stones = stones
        #self.frame_num = frame_num

    def __str__(self):
        self.order_stones()
        ret = "Stone positions             :\tColor\t\tX(ft)\t\tY(ft)\n"
        ret += ""
        for stone in self.stones:
            ret+="{}\n".format(stone)
        return ret

    def order_stones(self):
        if len(self.stones)==0: return # No rocks visible
        dists = np.empty(len(self.stones))
        ostones = []
        for i,stone in enumerate(self.stones):
            dists[i] = math.sqrt(math.pow(stone.x,2) + math.pow(stone.y,2))-self.stone_radius
        inds = np.argsort(dists)
        for i in inds:
            ostones.append(self.stones[i])
        self.stones = ostones
        return

    def score_layout(self):
        if len(self.stones)==0: return ('red',0) # No rocks visible
        dists = np.empty(len(self.stones))
        for i,stone in enumerate(self.stones):
            dists[i] = math.sqrt(math.pow(stone.x,2) + math.pow(stone.y,2))-self.stone_radius
            #print(dists[i])
        inds = np.argsort(dists)
        if dists[inds[0]] > 6.: return ('red',0) # No rocks on paint
        score = 1
        cprev = self.stones[inds[0]].color
        for ind in inds[1:]:
            if dists[ind] > 6.: return(cprev,score) # No more rocks on paint
            c = self.stones[ind].color
            if c==cprev: score += 1
            else: return (cprev,score)  # This is the other color, stop scoring
        return (cprev,score)  # No other stones to examine
    
    def stone_stats(self):
        count_red=0
        count_yel=0
        for stone in self.stones:
            if stone.color=='yellow': count_yel+=1
            elif stone.color=='red' : count_red+=1
        return count_red,count_yel

    def draw(self):
        h=1500 # height represents 30 ft in real world, everything scales to this
        w=int(h*14./30)  # width represents 14 ft in the read world
        ft2px = h/30.

        patches = []
        # Draw House + lines
        xc = int(w/2.)-1
        yc = int(h*22./30.)-1
        r12 = int(w*12./14./2.)
        r8 = int(r12*8./12.)
        r4 = int(r12*4./12.)
        rb = int(r12*1.3/12.)
        patches.append(Circle((xc, yc), r12 ,color='skyblue'))
        patches.append(Circle((xc, yc), r8 ,color='white'))
        patches.append(Circle((xc, yc), r4 ,color='red'))
        patches.append(Circle((xc, yc), rb ,color='white'))

        tl = Rectangle((0,yc),w,1,color='black')
        patches.append(tl)
        bl = Rectangle((0,int(h-h*2./30.)),w,1,color='black')
        patches.append(bl)
        hl = Rectangle((0,100),w,1,color='black')
        patches.append(hl)
        cl = Rectangle((xc,0),1,h,color='black')
        patches.append(cl)

        # Draw Stones
        for stone in self.stones:
            #print(stone)
            patches.append(Circle((xc+stone.x*ft2px, yc-stone.y*ft2px), self.stone_radius*ft2px ,color='grey'))
            patches.append(Circle((xc+stone.x*ft2px, yc-stone.y*ft2px), self.stone_radius*ft2px*2./3. ,color=stone.color))

        # Display everything
        fig = plt.figure(figsize=(h/100.,w/100.))
        img = np.ones((h,w,3),dtype=int)
        plt.imshow(img)
        for patch in patches:
            fig.gca().add_patch(patch)
        plt.show()


# Highest level game results, containing information about each end
class MatchResult:
    def __init__(self,name='Unknown'):
        self.version = 20180812  # YYYYMMDD - Update when results on the same video or interface change
        self.name = name # Name of video file
        self.metadata = MatchMetadata()
        self.end_results = []
        self.red_score = -1
        self.yel_score = -1

    def add_end_result(self,end_result):
        self.end_results.append(end_result)
        # Update final score, assume time ordered
        self.red_score = self.end_results[-1].red_score
        self.yel_score = self.end_results[-1].yel_score

    def __str__(self):
        ret = "Match results for \'{}\':\n".format(self.name)
        ret +="Final Red Score {}, Yel Score {}\n".format(self.red_score,self.yel_score)
        for end in self.end_results:
            ret+="{}\n".format(end)
        return ret

    def draw(self):
        print("Match results for \'{}\':\n".format(self.name))
        print("Final Red Score {}, Yel Score {}\n".format(self.red_score,self.yel_score))
        for end in self.end_results:
            end.draw()

# Used to store meta data concerning the matchup, like who is playing and when 
class MatchMetadata:
    def __init__(self, date='Unknown', red_name='Unknown', yel_name='Unknown',
                 event_name='Unknown', gender='Unknown', other='Unknown'):
        self.date             = date  # Date that the match took place in YYYYMMDD format
        self.team_name_red    = red_name
        self.team_name_yel    = yel_name
        self.event_name       = event_name  # E.g. Pyeong Change Olympics
        self.gender           = gender  # Men's, Women's, or Mixed
        self.other_info       = other  # E.g. "Best gold medal match in history!"

    #def set_by_hand(self):


# Stores all stone positions throughout an end and scores read from the scoreboard after the end is over
class EndResult:
    def __init__(self,end_num =-1,red_score=-1,yel_score=-1):
        self.end_num = end_num 
        self.red_score = red_score
        self.yel_score = yel_score
        self.stone_layouts = [] # One layout for each stone thrown
        self.stones_left = []   # One layout for each stone thrown, a tuple of (n_red_left, n_yel_left)
        self.red_hammer = -1 # -1 if not known yet, 0 if yellow has the last rock, 1 if red has the last rock

    def add_stone_layout(self,layout):
        self.stone_layouts.append(layout)

    def add_stones_left(self,n_red,n_yel):
        self.stones_left.append((n_red,n_yel))

    def set_score(red_score,yel_score):
        self.red_score = red_score
        self.yel_score = yel_score

    def __str__(self):
        ret = "Results for End {}\n".format(self.end_num)
        ret += "Red Score: {}, Yellow Score: {}\n".format(self.red_score,self.yel_score)
        if   self.red_hammer==1: ret += "Red has the hammer\n"
        elif self.red_hammer==0: ret += "Yellow has the hammer\n"
        else              : ret += "Unknown who has the hammer...\n"
        for entry in zip(self.stones_left, self.stone_layouts):
            ret+="Red left: {}, Yel left: {}\n  {}\n".format(entry[0][0], entry[0][1], entry[1])
        return ret

    def draw(self):
        print("Results for End {}\n".format(self.end_num))
        print("Red Score: {}, Yellow Score: {}\n".format(self.red_score,self.yel_score))
        for entry in zip(self.stones_left, self.stone_layouts):
            print("Red left: {}, Yel left: {}\n  {}\n".format(entry[0][0], entry[0][1], entry[1]))
            entry[1].draw()

# Basic container to hold game status data: current_end,current_score_red,current_score_yel,n_red_left,n_yel_left
class ScoreboardState:
    def __init__(self,current_end=-1,current_score_red=-1,current_score_yel=-1,n_red_left=-1,n_yel_left=-1):
        self.current_end = int(current_end)
        self.current_score_red = int(current_score_red)
        self.current_score_yel = int(current_score_yel)
        self.n_red_left = int(n_red_left)
        self.n_yel_left = int(n_yel_left)

    def valid(self):
        if (self.current_end==-1 or 
            self.current_score_red==-1 or self.current_score_yel==-1 or 
            self.n_red_left==-1 or self.n_yel_left==-1):
            return False
        return True

    def __str__(self):
        return "End: {}, Red Score: {}, Yellow Score: {}, Red Stones Left: {}, Yellow Stones Left: {}".format(
              self.current_end,
              self.current_score_red,
              self.current_score_yel,
              self.n_red_left,
              self.n_yel_left)
        

# Keeps a record of the last few frames, used to determine key frames and game flow
class ShortTermMemory:
    def __init__(self,cache_size=15):
        self.verbose = True
        self.max_lookback = cache_size
        self.frame_cache   = deque(maxlen=self.max_lookback)
        self.dframe_cache  = deque(maxlen=self.max_lookback)
        self.layout_cache  = deque(maxlen=self.max_lookback)
        self.sbstate_cache = deque(maxlen=self.max_lookback)

        self.current_end_result = EndResult()  # This gets filled in with information as it comes
        self.end_prev = 1
        self.n_red_prev = 8
        self.n_yel_prev = 8

    def clear_caches(self):
        self.frame_cache.clear()
        self.dframe_cache.clear()
        self.layout_cache.clear()
        self.sbstate_cache.clear()

    # Look through the last few cached frames, take the last frame with the most stones.
    # Stones can 'go missing' due to e.g. shadows or players walking and obstructing the view.
    # Use the last frame with all the stones because the stones could still be moving otherwise.
    # Returns an index of the best frame in the cache.
    def find_best_layout(self):
        if self.layout_cache is None or self.layout_cache == []: return -1 # empty layouts
        imax,nstone_max = -1,0
        for i,layout in enumerate(self.layout_cache):
            if layout is None: continue
            nred,nyel = layout.stone_stats()
            nstone = nred+nyel
            if nstone >= nstone_max: imax,nstone_max = i,nstone
        return imax

    # Look through the last few frames and track how many stone icons are remaining.  Take the minimum.  Note that
    # while a stone is being thrown, the display toggles back and forth for a length of time.  E.g. if the display
    # shows red5,yel5; red5,yel4; red5,yel5; red5,yel4; this means yel's 5th-to-last rock is currently being thrown.
    # The method should return red5,yel4 in this case.
    def find_nstones_stats(self):
        n_red_min,n_yel_min = 9,9
        for sbs in self.sbstate_cache:
            if sbs.n_red_left < n_red_min: n_red_min = sbs.n_red_left
            if sbs.n_yel_left < n_yel_min: n_yel_min = sbs.n_yel_left
        return n_red_min,n_yel_min

    def __str__(self):
        ret = 'ShortTermMemory state:\n'
        ret += 'Number of frames in cache: {}'.format(len(frame_cache))
        for layout,state in zip(self.layout_cache,self.sbstate_cache):
            ret+="{}\n".format(state)
            ret+="{}\n".format(layout)
        return ret


# Draw the house, set the height in pixels and everything scales to that.  Height represents 30 ft in real world.
# Returns patches list and the feet-to-pixel conversion factor, and pixel coordinate for the house center.
def draw_house(height=1500):
    h=height # height in pixels that matches 30 ft in real world, everything scales to this
    w=int(h*14./30.)  # width represents 14 ft in the real world
    ft2px = h/30.

    patches = []
    # Draw House
    xc = int(w/2.)-1
    yc = int(h*22./30.)-1
    r12 = int(w*12./14./2.) # 12-ft
    r8 = int(r12*8./12.)    # 8-ft
    r4 = int(r12*4./12.)    # 4-ft
    rb = int(r12*1.3/12.)   # Button
    patches.append(Circle((xc, yc), r12 ,color='skyblue'))
    patches.append(Circle((xc, yc), r8 ,color='white'))
    patches.append(Circle((xc, yc), r4 ,color='red'))
    patches.append(Circle((xc, yc), rb ,color='white'))

    # Draw lines
    tl = Rectangle((0,yc),w,1,color='black')              # T-line
    bl = Rectangle((0,int(h-h*2./30.)),w,1,color='black') # Back line
    hl = Rectangle((0,int(h*2./30.)),w,1,color='black')   # Hog line
    cl = Rectangle((xc,0),1,h,color='black')              # Center line
    patches.append(tl)
    patches.append(bl)
    patches.append(hl)
    patches.append(cl)
    
    return patches,ft2px,xc,yc

