"""
Test problems for Project 2 -- Dana Nau, Oct. 16, 2019

Here's a modified version of the test problems from Project 1. The changes are as
follows:

1. I removed or modified the ones that were obviously unsolvable.

2. Each problem is a list of the form [name, p0, finish, walls], where
   name is the name of the problem (for use as the title of the graphics window),
   p0 is the starting point,
   finish is the finish line,
   walls is a list of walls.

3. If a problem's dimensions are so small that the problem is unsolvable, you can
call double(p) where p is the problem, to return a problem in which the x and y
dimensions are both doubled.
"""


def double(problem):
	return [problem[0], double_point(problem[1]), double_edge(problem[2]), double_edges(problem[3])]

def double_point(point):
	return (2*point[0], 2*point[1])

def double_edge(edge):
	return [double_point(edge[0]), double_point(edge[1])]

def double_edges(edges):
	return [double_edge(e) for e in edges]



# A small rectangular region, with no obstacles in the way. There are four
# different orientations of the starting point and finish line, so you can
# check whether you're handling your x and y coordinates consistently.

rect20a = ['rect20a', (5,12), [(15,5),(15,15)],
	[[(0,0),(20,0)], [(20,0),(20,20)], [(20,20),(0,20)], [(0,20),(0,0)]]]

rect20b = ['rect20b', (15,12), [(5,5),(5,15)],
	[[(0,0),(20,0)], [(20,0),(20,20)], [(20,20),(0,20)], [(0,20),(0,0)]]]

rect20c = ['rect20c', (12,5), [(5,15),(15,15)],
	[[(0,0),(20,0)], [(20,0),(20,20)], [(20,20),(0,20)], [(0,20),(0,0)]]]

rect20d = ['rect20d', (12,15), [(5,5),(15,5)],
	[[(0,0),(20,0)], [(20,0),(20,20)], [(20,20),(0,20)], [(0,20),(0,0)]]]


# A larger rectangular region, with the starting point farther from the finish
# line. This is useful for testing whether you're overshooting.

rect50 = ['rect50', (3,40), [(30,10),(30,30)],
	[[(0,0),(50,0)], [(50,0),(50,50)], [(50,50),(0,50)], [(0,50),(0,0)]]]

rect100 = ['rect100', (6,80), [(60,20),(60,60)],
	[[(0,0),(100,0)], [(100,0),(100,100)], [(100,100),(0,100)], [(0,100),(0,0)]]]

# the finish line is behind a wall

wall16a = ['wall16a', (4, 2), [(8, 10), (10, 10)], 
	[[(0, 0), (16, 0)], [(16, 0), (16, 16)], [(16, 16), (0, 16)], [(0, 16), (0, 0)], [(8, 0), (8, 10)]]]

wall16b = ['wall16b', (4, 2), [(8, 2), (10, 2)], [[(0, 0), (16, 0)], [(16, 0), (16, 16)], [(16, 16), (0, 16)], [(0, 16), (0, 0)], [(8, 0), (8, 10)]]]


lhook32 = ['lhook32', (4, 2), [(26, 2), (28, 2)],
	[((0, 0), (32, 0)), ((32, 0), (32, 32)), ((32, 32), (0, 32)), ((0, 32), (0, 0)), ((18, 0), (18, 22)), ((18, 22), (12, 22)), ((12, 22), (12, 14))]]


rhook32a = ['rhook32a', (4, 4), [(20, 14), (22, 14)],
	[[(0, 0), (32, 0)], [(32, 0), (32, 32)], [(32, 32), (0, 32)], [(0, 32), (0, 0)], [(8, 0), (8, 24)], [(8, 24), (24, 24)], [(24, 24), (24, 10)]]]


rhook32b = ['rhook32b', (4, 4), [(20, 14), (22, 14)], 
	[[(0, 0), (32, 0)], [(32, 0), (32, 32)], [(32, 32), (0, 32)], [(0, 32), (0, 0)], [(8, 0), (8, 24)], [(8, 24), (24, 24)], [(24, 24), (24, 8)]]]


# a spiral-shaped wall

spiral32 = ['spiral32', (4, 4), [(20, 12), (22, 12)], [[(0, 0), (32, 0)], [(32, 0), (32, 32)], [(32, 32), (0, 32)], [(0, 32), (0, 0)], [(8, 0), (8, 24)], [(8, 24), (24, 24)], [(24, 24), (24, 8)], [(24, 8), (16, 8)], [(16, 8), (16, 16)]]]


# the example in the project description, and a variant with the wall farther down

pdes30 = ['pdes30', (4,5), [(24,8),(26,8)],
	[[(0,0),(10,0)], [(10,0),(10,10)], [(10,10),(20,10)], [(20,10),(30,0)], [(30,0),(30,10)], [(30,10),(10,20)], [(10,20),(0,20)], [(0,20),(0,0)], [(3,14),(10,14)], [(10,14),(10,16)], [(10,16),(3,16)], [(3,16),(3,14)]]]

pdes30b = ['pdes30b', (4,5), [(28,3),(29,3)],
	[[(0,0),(10,0)], [(10,0),(10,10)], [(10,10),(20,10)], [(20,10),(30,0)], [(30,0),(30,10)], [(30,10),(10,20)], [(10,20),(0,20)], [(0,20),(0,0)], [(3,14),(10,14)], [(10,14),(10,16)], [(10,16),(3,16)], [(3,16),(3,14)]]]

# the finish line is behind a wall that's connected to a rectangular obstacle

rectwall32 = ['rectwall32', (12,4), [(20,4),(24,4)],
	[[(0,0),(32,0)], [(32,0),(32,32)], [(32,32),(0,32)], [(0,32),(0,0)], [(16,0),(16,8)], [(12,8),(20,8)], [(20,8),(20,24)], [(20,24),(12,24)], [(12,24),(12,8)]]]

rectwall32a = ['rectwall32a', (12,4), [(20,4),(24,4)],
    [[(0,0),(32,0)], [(32,0),(32,32)], [(32,32),(0,32)], [(0,32),(0,0)], [(16,0),(16,8)], [(12,8),(20,8)], [(20,8),(20,22)], [(20,22),(12,22)], [(12,22),(12,8)], [(16,32),(16,24)]]]

# two walls

walls32 = ['walls32', (12,4), [(20,4),(24,4)],
	[[(0,0),(32,0)], [(32,0),(32,32)], [(32,32),(0,32)], [(0,32),(0,0)], [(16,0),(16,8)], [(12,8),(16,8)], [(12,22),(12,8)], [(20,32),(20,16)]]]


# twisty0 cut in half and widened

twisty1 = \
['twisty1', (60, 26), [(2, 4), (7, 4)], [[(6, 1), (10, 7)], [(10, 7), (14, 8)], [(14, 8), (17, 8)], [(17, 8), (20, 9)], [(20, 9), (26, 12)], [(26, 12), (27, 11)], [(27, 11), (27, 10)], [(27, 10), (26, 4)], [(26, 4), (27, 2)], [(27, 2), (29, 0)], [(29, 0), (34, 0)], [(34, 0), (43, 1)], [(43, 1), (47, 1)], [(47, 1), (52, 2)], [(52, 2), (56, 4)], [(56, 4), (57, 6)], [(57, 6), (57, 10)], [(57, 10), (56, 12)], [(56, 12), (54, 16)], [(54, 16), (55, 18)], [(55, 18), (58, 21)], [(58, 21), (62, 24)], [(62, 24), (62, 29)], [(55, 29), (56, 28)], [(56, 28), (56, 26)], [(56, 26), (54, 24)], [(54, 24), (51, 22)], [(51, 22), (47, 20)], [(47, 20), (44, 18)], [(44, 18), (44, 16)], [(44, 16), (48, 14)], [(48, 14), (50, 11)], [(50, 11), (51, 9)], [(51, 9), (51, 8)], [(51, 8), (50, 7)], [(50, 7), (46, 6)], [(46, 6), (40, 8)], [(40, 8), (37, 8)], [(37, 8), (34, 6)], [(34, 6), (32, 5)], [(32, 5), (32, 8)], [(32, 8), (33, 11)], [(33, 11), (33, 14)], [(33, 14), (27, 18)], [(27, 18), (22, 17)], [(22, 17), (19, 15)], [(19, 15), (17, 14)], [(17, 14), (14, 14)], [(14, 14), (8, 13)], [(8, 13), (4, 9)], [(4, 9), (1, 4)], [(1, 4), (0, 3)], [(0, 3), (3, 2)], [(3, 2), (6, 1)], [(6, 1), (6, 1)], [(62, 29), (55, 29)]]]
