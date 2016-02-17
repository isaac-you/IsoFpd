Codes for paper "A novel community detection method based on density 
clustering algorithm" by Tao You and Zhong-Yuan Zhang.

To run the tests, just load the files and run demo.m in Matlab.

All codes are tested under Matlab Version 2015b 64bit and Matlab 
Version 2010a 32bit

This document contains following files:

1.IsoFdp.m 			%the main Codes of the proposed algorithm

2.IsomapII.m 		%faster version of isomap

3.dijkstra.mexw64 	%64bit compiled version of dijkstra algorithm which is 
much faster than the Floyd's algorithm
4.dijkstra.mexw32   %32bit compiled version of dijkstra algorithm

5.demo.m 			%% show how to use the algorithm

6.LFR_data 			%% the synthetic dataset to test the algorithm

7.readme

Note: The codes from line 31 to line 72 in IsoFdp.m is changed. By utilizing 
sparse matrix and custom function in pdist, this version of 'structure 
similarity' is about 100x faster than the original one which is coded by 
double for-loop.  

Feedback is very welcome. If you have found a bug, or have any problems or 
commnets, please write to: 
isaacyou@email.cufe.edu.cn;
isaactyou@gmail.com;
zhyuanzh@gmail.com

02/13/2016

Have fun!

%%------------------------------------------------
