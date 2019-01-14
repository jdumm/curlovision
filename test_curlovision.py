import os
import glob
import pickle
import curlovision as cvis
import datetime

filename = 'data/samples/vlc-record-2018-06-06-17h35m21s-PyeongChang.Olympics.2018.curling.men.mp4'
print('Test #1 commencing.  It should take ~4 minutes.')
print('Starting datetime:', datetime.datetime.now())
frame_start=0
print('Processing video file: {}'.format(filename))
match_result_new = cvis.process_video(filename,frame_start=frame_start,
                                  icon_filenames=['data/stone_icons/red_icon_Olympics2018.npy','data/stone_icons/yel_icon_Olympics2018.npy'],
                                  draw_key_frames=False,draw_test_frames=False,
                                  debug_12ft=False,debug_stones=False,debug_scoreboard=False)
print('Ending   datetime:', datetime.datetime.now())

rebuild_test_result = False # Set to True to reset test case comparison, in case things should change

compare_filename = 'data/samples/' + str(os.path.basename(filename))+'.pkl'
if rebuild_test_result:
    with open(compare_filename, 'wb') as f:
        pickle.dump(match_result_new,f)
    print('Finished rebuilding test!')
else:
    with open(compare_filename, 'rb') as f:
        match_result_old = pickle.load(f)
    if (match_result_new == match_result_old):
	    print('Test #1 Success: Reproduced the same match results!')
    else:
	    print('Test #1 Failure: Difference found in match results!')

