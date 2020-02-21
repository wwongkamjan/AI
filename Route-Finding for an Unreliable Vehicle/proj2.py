"""

I pledge on my honor that I have not given or received any unauthorized assistance on this project.

Wichayaporn Wongkamjan

In this project, I mainly come up with some conditions to handle with crashing opponent (opponent1) and modify h_function (h_walldist).

"""

import racetrack_example as rt
import math
import sys
import json
import opponents as op

# Global variable for h_walldist
infinity = float('inf')     # same as math.inf

# Your proj2 function
def main(state,finish,walls):
    """
    main function is to calculate for the best velocity of this current state. We put our solution in choices.txt.
    First, I added the current state to visited state list (using data2.txt)
    Second, I checked for the current state
    if it's the goal and velocity (u,v) is in range of (0,0) then the best velocity is (0,0) and we win.
    If not, we're going to look for the best velocity by searching for the best next state and velocity.
    Third, we might found some velocity that make us progress in a loop, we will avoid that checking if the next state is visited.
    Lastly, we return our best velocity choice.
    """
    ((x,y), (u,v)) = state

    
    # Retrieve the grid data that the "initialize" function stored in data.txt
    data_file = open('data.txt', 'r')
    grid = json.load(data_file)
    data_file.close()
    
    
    choices_file = open('choices.txt', 'w')
    
    # Add visited state to the file data2.txt
    if (u,v) == (0,0) and edist_to_line((x,y),finish) > 1:
        data_file = open('data2.txt', 'w')
        visited = []
        visited.append((x,y))
        json.dump(visited,data_file)
        data_file.close()
    else:
        data_file = open('data2.txt', 'r')
        visited = json.load(data_file)
        data_file.close()
        data_file = open('data2.txt', 'w')
        visited.append((x,y))
        json.dump(visited,data_file)
        data_file.close()
        
    # Take the new version of h_walldist, which needs the grid as a 4th argument, and
    # translate it into the three-argument function needed by rt.main
    h = lambda state,fline,walls: h_walldist(state,fline,walls,grid)    

    # We checked if this current state can be the end
    if edist_to_line((x,y),finish) <= 1 and abs(u) <= 2 and abs(v) <= 2:
        velocity = (0,0)
        print('  proj2_example: finishing, new velocity =', velocity)
        
    # We search for the best velocity
    else:
        path = rt.main(state,finish,walls,'gbf', h, verbose=0, draw=0)
        #print('  proj2_example: path =', path)
        for i in range(len(path)):
            #print('  proj2_example: path[',i,'] =', path[i])
            if path[i] == state:
                print('  proj2_example: found state', state)
                velocity = path[i+1][1]
                
                # the case to handle if the velocity will lead us to crashing
                ebest = op.opponent1((x,y),(velocity[0],velocity[1]),finish,walls)
                if rt.crash(((x,y),(x+velocity[0]+ebest[0],y+velocity[1]+ebest[1])),walls):
                    velocity = (velocity[0]-ebest[0],velocity[1]-ebest[1])
                
                data_file = open('data2.txt', 'r')
                visited = json.load(data_file)
                data_file.close()
                
                # the case to handle if the velocity will lead us to visited states
                if visited is not None:
                    ebest = op.opponent1((x,y),(velocity[0],velocity[1]),finish,walls)
                    if [x+velocity[0]+ebest[0],y+velocity[1]+ebest[1]] in visited:
                        velocity = (velocity[0]-ebest[0],velocity[1]-ebest[1])
                        ebest = op.opponent1((x,y),(velocity[0],velocity[1]),finish,walls)

                print('  proj2_example: new velocity', velocity)
                break
            
    # need to flush because Python uses buffered output
    print(velocity,file=choices_file,flush=True)

def edist_to_line(point, edge):
    """
    Euclidean distance from (x,y) to the line ((x1,y1),(x2,y2)).
    """
    (x,y) = point
    ((x1,y1),(x2,y2)) = edge
    if x1 == x2:
        ds = [math.sqrt((x1-x)**2 + (y3-y)**2) \
            for y3 in range(min(y1,y2),max(y1,y2)+1)]
    else:
        ds = [math.sqrt((x3-x)**2 + (y1-y)**2) \
            for x3 in range(min(x1,x2),max(x1,x2)+1)]
    return min(ds)
                

def initialize(state,fline,walls):    
    """
    Call edist_grid to initialize the grid for h_walldist, then write the data, in
    json format, to the file "data.txt" so it won't be lost when the process exits
    """
    edist_grid(fline,walls)
    data_file = open('data.txt', 'w')
    json.dump(grid,data_file)
    data_file.close()

   # print(grid)


def h_walldist(state, fline, walls, grid):
    """
    The new version of h_walldist no longer calls edist_grid, but instead takes
    the grid as a fourth argument. It retrieves the current position's grid value,
    and adds an estimate of how long it will take to stop. 
    """

    ((x,y),(u,v)) = state
    hval = float(grid[x][y])   
    # if this state is going to lead us to crashing, then no.
    ebest = op.opponent1((x,y),(u,v),fline,walls)
    if rt.crash(((x,y),(x+u+ebest[0],y+v+ebest[1])),walls):
        hval = infinity    
   
    # add a small penalty to favor short stopping distances
    au = abs(u); av = abs(v); 
    sdu = au*(au-1)/2.0
    sdv = av*(av-1)/2.0
    sd = max(sdu,sdv)
    penalty = sd/10.0

    # compute location after fastest stop, and add a penalty if it goes through a wall
    if u < 0: sdu = -sdu
    if v < 0: sdv = -sdv
    sx = x + sdu
    sy = y + sdv
    if rt.crash([(x,y),(sx,sy)],walls):
        hval = infinity
    #hval = max(hval+penalty,sd)


    return hval


def edist_grid(fline,walls):
    global grid
    xmax = max([max(x,x1) for ((x,y),(x1,y1)) in walls])
    ymax = max([max(y,y1) for ((x,y),(x1,y1)) in walls])
    grid = [[edistw_to_finish((x,y), fline, walls) for y in range(ymax+1)] for x in range(xmax+1)]
    flag = True
    print('computing edist grid', end=' '); sys.stdout.flush()
    while flag:
        print('.', end=''); sys.stdout.flush()
        flag = False
        for x in range(xmax+1):
            for y in range(ymax+1):
                for y1 in range(max(0,y-1),min(ymax+1,y+2)):
                    for x1 in range(max(0,x-1),min(xmax+1,x+2)):
                        if grid[x1][y1] != infinity and not rt.crash(((x,y),(x1,y1)),walls):
                            if x == x1 or y == y1:
                                d = grid[x1][y1] + 1
                            else:
                                # In principle, it seems like a taxicab metric should be just as
                                # good, but Euclidean seems to work a little better in my tests.
                                d = grid[x1][y1] + 1.4142135623730951
                            if d < grid[x][y]:
                                grid[x][y] = d
                                flag = True
    print(' done')
    return grid


def edistw_to_finish(point, fline, walls):
    """
    straight-line distance from (x,y) to the finish line ((x1,y1),(x2,y2)).
    Return infinity if there's no way to do it without intersecting a wall
    """
#   if min(x1,x2) <= x <= max(x1,x2) and  min(y1,y2) <= y <= max(y1,y2):
#       return 0
    (x,y) = point
    ((x1,y1),(x2,y2)) = fline
    # make a list of distances to each reachable point in fline
    if x1 == x2:           # fline is vertical, so iterate over y
        ds = [math.sqrt((x1-x)**2 + (y3-y)**2) \
            for y3 in range(min(y1,y2),max(y1,y2)+1) \
            if not rt.crash(((x,y),(x1,y3)), walls)]
    else:                  # fline is horizontal, so iterate over x
        ds = [math.sqrt((x3-x)**2 + (y1-y)**2) \
            for x3 in range(min(x1,x2),max(x1,x2)+1) \
            if not rt.crash(((x,y),(x3,y1)), walls)]
    ds.append(infinity)    # for the case where ds is empty
    return min(ds)

            
            
            
    
