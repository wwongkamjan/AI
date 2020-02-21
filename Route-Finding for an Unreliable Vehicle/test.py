# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:09:14 2019

@author: wwong
"""


import racetrack_example as rt
import sample_probs as sp
import sys
import math
import env
import opponents as op
import json
import time
import multiprocessing as mp


env.main(problem=sp.lhook32  , max_search_time=5, max_init_time=5, opponent=op.opponent1, verbose=1, draw=1)
#time.sleep(5)
#p.initialize((5,12),[(15,5),(15,15)],[[(0,0),(20,0)], [(20,0),(20,20)], [(20,20),(0,20)], [(0,20),(0,0)]])
    
    #rt.main(sp.rect20, "gbf", sh.h_walldist ,sh, draw=1, verbose=1)
 #rt.main(sp.rectwall16, "gbf", p.h_proj1 ,p , draw=1, verbose=1)
 
 
#p = mp.Process(target=proj2.initialize, \
               #args = (((4, 4),(0,0)), [(20, 14), (22, 14)], [[(0, 0), (32, 0)], [(32, 0), (32, 32)], [(32, 32), (0, 32)], [(0, 32), (0, 0)], [(8, 0), (8, 24)], [(8, 24), (24, 24)], [(24, 24), (24, 8)]], ))
#p.start()
# Wait for max_init_time seconds
#p.join(1000)

#proj2.initialize(((4, 4),(0,0)), [(20, 14), (22, 14)], [[(0, 0), (32, 0)], [(32, 0), (32, 32)], [(32, 32), (0, 32)], [(0, 32), (0, 0)], [(8, 0), (8, 24)], [(8, 24), (24, 24)], [(24, 24), (24, 8)]])