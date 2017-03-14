import subprocess

import numpy as np

class VideoSink(object) :

    def __init__( self, size, filename="output", rate=10, byteorder="bgra" ) :
            self.size = size
            cmdstring  = ('mencoder',
                    '/dev/stdin',
                    '-demuxer', 'rawvideo',
                    '-rawvideo', 'w={}:h={}'.format(*size[::-1])+":fps={}:format={}".format(rate,byteorder),
                    '-o', filename+'.avi',
                    '-ovc', 'lavc',
                    )
            self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def run(self, image) :
            *assert image.shape == self.size
            self.p.stdin.write(image.tostring())
    def close(self) :
            self.p.stdin.close()
