# Curlovision
Curling computer vision project for extracting stone positions from broadcast game footage.  The purpose is to build a data set of stone positions in high level curling matches, to be used in generating predictive models of the score or next stone positions.

![Highlight Slide](example_images/curlovision_slide.png)

# Requirements
Curlovision is written in Python (tested with v3.6) taking advantage of the following dependencies:

  * OpenCV
  * Numpy
  * Pytesseract
  * Jupyter
  * Matplotlib

There is a docker available with all required dependencies, available thanks to [zbassett](https://github.com/zbassett) at the following link:
[https://github.com/zbassett/jupyterlab_opencv](https://github.com/zbassett/jupyterlab_opencv)

# Usage

Main functions/classes can be imported from 'curlovision.py'.  Example files below are Jupyter Notebooks.

'ProcessBatch.ipynb' is an example for processing video broadcasts of curling matches.
  * Video files must be in mp4 format, 1 match per file.
  * Video assumed to have 15 fps with resolution 1920x1080.  Higher framerates can be processed efficiently with higher nskip values.
  * Every 5th frame (default) is searched for the blue ice of the 12ft ring in the house.  When found, a ShortTermMemory starts collecting frames until the blue ice disappears.  Key frames are identified and processed.  The whole process happens roughly 2x faster than video play time.  Hough Transforms are used to quickly find rings of the house and stones.  The scoreboard and games status are read and recorded using pytesseract and OpenCV template matching.  
  * The process_video function returns a custom MatchResult object that contains the results from each end of the match.  Each EndResult contains the stone positions in feet after each stone is thrown.  The number of stones remaining to be thrown by red and yellow teams are also recorded, along with the team that has the hammer (last stone in the end).
  * MatchResults know how to .draw() themselves.

'BasicVisualizeExamples.ipynb' is an example for parsing and visualizing the MatchResults.  It was used to generate the example heatmap image on the left:

![Example Heatmap](example_images/ExampleHeatmapImage.png)
![Example Guard Heatmap](example_images/ExampleFreeGuardImage.png)

'FreeGuardAnalysis.ipynb' is a more sophisticated case for analyzing the match outcomes based on the first 4 stones and generated the heatmap above on the right.

'VideoDebugSandbox.ipynb' can be used to visualize and fix all the subprocesses in video processing.

# TODO Wishlist

Here are the next steps I would like to take, roughly in order of priority:
  * Abstract out all tournament-specific parameters to make them easily configurable: Scoreboard location, stone icons, house and stone color space.
  * Provide a ProcessFrame.ipynb example notebook to illustrate how to handle a single key frame.
  * Track down and fix scoreboard reading errors (2 known).
  * Explore ways to import/process matches available on Youtube.

# Common Failures

Curlovision is not perfect.  Testing suggests typical stone positions are accurate to within 2 or 3 inches.  Perhaps 1 or 2% of StoneLayouts are off by ~10%, due to players/stones obstructing too much of the house and ruining the measurement metrics.  There are occassionaly false positive stones (e.g. matching a round head with a red hat).  Also, sometimes stones on the edge of the house can still be in motion, on their way completely out as the camera view changes.  Stones with ~20% occlusion from players and brooms can be overlooked.  

There may not be a StoneLayout for every stone thrown, especially in cases where the broadcast footage does not provide a clean overhead view of the house after, e.g., inconsequential stones.  Often the very last stone thrown is not shown during broadcast if the outcome is already clear.  Very rarely the scoreboard reading can fail and produce random results.  

Various levels of debugging can be enabled during process_video to examine the performance more closely.


# Open Source

Curolovision is released under a BSD license and is free for both academic and commercial use.
