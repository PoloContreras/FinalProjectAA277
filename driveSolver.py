import numpy as np
import cvxpy as cvx
from copy import deepcopy

#HELPER FUNCTIONS FOR DYNAMICS CONSTRAINTS
def nxt(var: cvx.Variable): #extract all elements of the array except the zeroth
    return var[1:]
def curr(var: cvx.Variable): #extract all elements of the array except the last
    return var[:-1]

#HELPER FUNCTION TO CREATE AN INITIAL GUESS FOR VEHICLE START
def initializer(xInitial,xFinal,maxX,minX,maxY,minY,zTraj,rTotal,time,freq,roboRadius,conservative=False): #initialize with a trajectory that avoids obstacles but ignores dynamics constraints
    
    #DECLARATION OF VALUES FOR NAVIGATION
    # h = 1./freq #size of time steps (reciprocal of frequency of instructions in hertz)
    NavigatioN = int(time*freq) #total number of steps in navigation: NavigatioN * h gives the number of seconds in the simulated environment

    if conservative:
        return np.repeat(np.reshape(xInitial,(1,np.shape(xInitial)[0])),NavigatioN,axis=0) #constant repetitions of original position as trajectory initialization, for reliability #constant repetitions of original position as trajectory initialization: other tricks didn't work

    #INTERMEDIATE VALUES TAKEN FROM ENVIRONMENT CHARACTERISTICS
    obstacles = np.shape(rTotal)[0] #total number of obstacles to dodge

    xt = np.linspace(xInitial,xFinal,NavigatioN) #optimal solution when disregarding both obstacles and dynamic constraints (from calculus of variations, just trust me)

    print('Initializing trajectory solver.')

    #INITIALIZATION OF LOOP CONTROL VALUES
    diff = np.inf #initialize to unreasonable value to overwrite in loop
    epsilon = 1e-3 #tolerance for convergence of the solution (solver with dynamics uses 1e-3)

    optval = np.inf

    iter = 0

    while abs(diff) > epsilon and iter <= 150:

        #SOLVING TRAJECTORY AS CONVEX PROBLEM
        x = cvx.Variable((NavigatioN,5)) #state trajectory

        #OBJECTIVE FUNCTION INITIALIZATION
        cost = cvx.square(cvx.norm(nxt(x) - curr(x),'fro')) #sum of squared differences creates a smooth trajectory from start position to target state

        constrNav = [] #initialize constraints list

        #INITIAL POSITION
        constrNav += [x[0,:] == xInitial] #hard constraint on initial position
        constrNav += [x[NavigatioN-1,:] == xFinal] #hard constraint on final position

        #WORK SPACE CONSTRAINTS
        constrNav += [x[:,1] >= minY + roboRadius, x[:,1] <= maxY - roboRadius]
        constrNav += [x[:,0] >= minX + roboRadius, x[:,0] <= maxX - roboRadius]

        #OBSTACLE AVOIDANCE CONSTRAINTS
        zt = xt[:,:2] #saving only the positional values in the trajectory (for obstacle avoidance)
        zDiffs = cvx.Parameter((NavigatioN,2))
        zSub = np.transpose(np.repeat(np.expand_dims(zt,axis=2),obstacles,axis=2),axes=(2,0,1)) \
            - np.transpose(zTraj,axes=(1,0,2)) #OxNx2 tensor containing differences between each position in the trajectory and obstacle center
        zSubNorms = np.linalg.norm(zSub,ord=2,axis=(-1))#OxN matrix of Euclidean distances between each position in the trajectory and each obstacle center
        zDiffs = x[:,:2] - zt #difference between current and prior position trajectory

        # Soft constraints on obstacle avoidance
        # Encourage the solver to have a margin of error in avoiding obstacles
        xSub = zSub[:,:,0]
        ySub = zSub[:,:,1]
        xDiffs = zDiffs[:,0]
        yDiffs = zDiffs[:,1]

        for o in range(obstacles):
            constrNav += [
                rTotal[o,:] \
                - zSubNorms[o,:] \
                - (cvx.multiply(xSub[o,:], xDiffs) + cvx.multiply(ySub[o,:], yDiffs)) / zSubNorms[o,:] <= 0
                ]

        objectNav = cvx.Minimize(cost)

        #DEFINE AND SOLVE CONVEX PROBLEM
        problem = cvx.Problem(objectNav,constrNav)
        optval = problem.solve(solver=cvx.ECOS)
        #END OF CONVEX OPTIMIZATION PART

        print('initializer opt :',optval) #monitor
        print('initializer problem status: ',problem.status)

        if problem.status == 'infeasible': #if bad information from the environment provokes an infeasible solve, don't crash
            conservative = True

        diff = np.max(np.abs(x.value - xt)) #for checking convergence of trajectory solution
        xt = deepcopy(x.value) #for use in following iteration
        zt = x[:,:2] #ACTIVATE FOR OBSTACLE AVOIDANCE CONSTRAINTS
        # ut = deepcopy(u.value) #for use in following iteration
        print('initializer convergence measure: ',diff) #monitor
        iter += 1

    print('initializer optimal value: ',optval)

    path = copy.deepcopy(x.value) #once more with feeling
    # positions = copy.deepcopy(x[:,0:2].value) #positional values only

    return path

#HELPER FUNCTION TO PLAN VEHICLE TRAJECTORY BASED ON CURRENT STATE
def trajPlan(xInitial,xFinal,maxX,minX,maxY,minY,rObs,zTraj,time,freq,speed_limit,kappa_limit,roboRadius,maxIter):

    #DECLARATION OF VALUES FOR NAVIGATION
    h = 1./freq #size of time steps (reciprocal of frequency of instructions in hertz)
    NavigatioN = int(time*freq) #total number of steps in navigation: NavigatioN * h gives the number of seconds in the simulated environment

    #INTERMEDIATE VALUES TAKEN FROM ENVIRONMENT CHARACTERISTICS
    obstacles = np.shape(rObs)[0] #total number of obstacles to dodge
    rTotal = np.repeat(rObs,NavigatioN,axis=1) #repeated instances of the obstacle radii, one per time step, for obstacle avoidance

    xt = initializer(xInitial,xFinal,maxX,minX,maxY,minY,zTraj,rTotal,time,freq,roboRadius) #warm start, for speed

    print("start: ",xInitial,"\ntarget: ",xFinal)
    print("Total time elapsed in trajectory: ",time," seconds\nFrequency of instructions: ",freq," hertz\nTotal number of time steps in trajectory: ",NavigatioN)

    #INITIALIZATION OF LOOP CONTROL VALUES
    diff = np.inf #initialize to unreasonable value to overwrite in loop
    epsilon = 1e-3 #tolerance for convergence of the solution

    optval = np.inf

    ut = np.zeros((NavigatioN,2)) #arbitrary stand-in for previous solution

    iter = 0

    while abs(diff) > epsilon and iter <= maxIter:

        prev_theta = xt[:,2] #history of angles
        prev_veloc = xt[:,3] #history of velocities
        prev_kappa = xt[:,4] #history of curvatures

        #SOLVING TRAJECTORY AS CONVEX PROBLEM
        x = cvx.Variable((NavigatioN,5)) #state trajectory
        u = cvx.Variable((NavigatioN,2)) #control inputs

        #OBJECTIVE FUNCTION INITIALIZATION
        #cost = h*cvx.square(cvx.norm(u,'fro')) + cvx.norm(cvx.multiply((xFinal - x[-1,:]),np.array([10,10,1,1,1])),p=1) #minimize distance to goal state and control input magnitudes
        #cost = h*cvx.sum(cvx.huber(u,M=1)) + cvx.norm(cvx.multiply((xFinal - x[-1,:]),np.array([10,10,1,1,1])),p=1) #alternative with less emphasis on control input magnitudes
        cost = h*cvx.square(cvx.norm(u,'fro')) #minimize control input magnitudes
        cost -= cvx.norm(x[1:,3], p=np.inf) #maximize overall vehicle speed
        cost += cvx.norm(cvx.multiply((xFinal - x[-1,:]),np.array([10,10,1,1,1])),p=1) #minimize distance to target location

        constrNav = [] #initialize constraints list

        #INITIAL POSITION
        constrNav += [x[0,:] == xInitial] #hard constraint on initial position
        #constrNav += [x[NavigatioN-1,:] == xFinal] #hard constraint on final position

        #DYNAMICS CONSTRAINTS
        constrNav += [
            nxt(x[:,0]) == curr(x[:,0]) + h * ( #xpos constraint x[:,0]
                    cvx.multiply(curr(x[:,3]), np.cos(curr(prev_theta)))
                    - cvx.multiply(cvx.multiply(curr(prev_veloc), np.sin(curr(prev_theta))), curr(x[:,2] - prev_theta))
                ),
            nxt(x[:,1]) == curr(x[:,1]) + h * ( #ypos constraint x[:,1]
                    cvx.multiply(curr(x[:,3]), np.sin(curr(prev_theta)))
                    + cvx.multiply(cvx.multiply(curr(prev_veloc), np.cos(curr(prev_theta))), curr(x[:,2] - prev_theta))
                ),
            nxt(x[:,2]) == curr(x[:,2]) + h * ( #theta constraint x[:,2]
                    cvx.multiply(curr(x[:,3]), curr(prev_kappa))
                    + cvx.multiply(curr(prev_veloc), curr(x[:,4]) - curr(prev_kappa))
                ),
            nxt(x[:,3]) == curr(x[:,3]) + h * curr(u[:,0]), #velocity constraint x[:,3]
            nxt(x[:,4]) == curr(x[:,4]) + h * curr(u[:,1]), #kappa constraint x[:,4]
            ]

        #CONTROL LIMIT CONSTRAINTS
        constrNav += [
            cvx.norm(x[1:,3], p=np.inf) <= speed_limit, #max forwards/reverse velocity (speed limit)
            x[1:,4] <= kappa_limit, #maximum curvature
            x[1:,4] >= -1*kappa_limit,
            ]

        #ROAD LIMIT CONSTRAINTS
        constrNav += [x[:,1] >= minY + roboRadius, x[:,1] <= maxY - roboRadius]
        constrNav += [x[:,0] >= minX + roboRadius, x[:,0] <= maxX - roboRadius]

        #OBSTACLE AVOIDANCE CONSTRAINTS
        zt = xt[:,:2] #saving only the positional values in the trajectory (for obstacle avoidance)
        zDiffs = cvx.Parameter((NavigatioN,2))
        zSub = np.transpose(np.repeat(np.expand_dims(zt,axis=2),obstacles,axis=2),axes=(2,0,1)) \
            - np.transpose(zTraj,axes=(1,0,2)) #OxNx2 tensor containing differences between each position in the trajectory and obstacle center
        zSubNorms = np.linalg.norm(zSub,ord=2,axis=(-1))#OxN matrix of Euclidean distances between each position in the trajectory and each obstacle center
        zDiffs = x[:,:2] - zt #difference between current and prior position trajectory

        # Soft constraints on obstacle avoidance
        # Encourage the solver to have a margin of error in avoiding obstacles
        xSub = zSub[:,:,0]
        ySub = zSub[:,:,1]
        xDiffs = zDiffs[:,0]
        yDiffs = zDiffs[:,1]

        for o in range(obstacles):
            cost += 10 * cvx.sum(cvx.maximum( #obstacle avoidance constraints integrated to cost function
                rTotal[o,:] \
                - zSubNorms[o,:] \
                - (cvx.multiply(xSub[o,:], xDiffs) + cvx.multiply(ySub[o,:], yDiffs)) / zSubNorms[o,:]
                + 1,
                0
            ))
            # constrNav += [ #hard version of obstacle avoidance constraints
            #     rTotal[o,:] \
            #     - zSubNorms[o,:] \
            #     - (cvx.multiply(xSub[o,:], xDiffs) + cvx.multiply(ySub[o,:], yDiffs)) / zSubNorms[o,:] <= 0
            #     ]

        objectNav = cvx.Minimize(cost)

        #DEFINE AND SOLVE CONVEX PROBLEM
        problem = cvx.Problem(objectNav,constrNav)
        optval = problem.solve(solver=cvx.ECOS)
        #END OF CONVEX OPTIMIZATION PART

        print('opt :',optval) #monitor
        print('problem status: ',problem.status)

        if problem.status == 'infeasible': #if bad information from the environment provokes an infeasible solve, don't crash
            xt = initializer(xInitial,xFinal,maxY,minY,zTraj,rTotal,time,freq,roboRadius,True)
            print('infeasible solution, restarting...')
            iter = 0
        else: 
            diff = np.max(np.abs(u.value - ut)) #for checking convergence of trajectory solution
            xt = deepcopy(x.value) #for use in following iteration
            ut = deepcopy(u.value) #for use in following iteration
            print('convergence measure: ',diff) #monitor
            iter += 1
        zt = x[:,:2] #ACTIVATE FOR OBSTACLE AVOIDANCE CONSTRAINTS

    print('final optimal value: ',optval)

    path = copy.deepcopy(x.value) #once more with feeling
    positions = copy.deepcopy(x[:,0:2].value) #positional values only
    commands = copy.deepcopy(u.value) #matrix of actions to be taken by agent

    return path,positions,commands

# maxIter = 50 #maximum number of convex solves before moving on
# xInitial = np.array([[0],[0],[0],[0],[0]]) #Vector indicating the current vehicle position: position in x, position in y, angle with respect to world frame, velocity, reciprocal of turn radius
# xFinal = np.array([[0],[0],[0],[0],[0]]]) #Vector indicating the target vehicle position: same variables as above
# minX = -20 #lower limit of navigable space in x direction
# maxX = 20 #upper limit of navigable space in x direction
# maxY = 20 #upper limit of navigable space in y direction
# minY = -20 #lower limit of navigable space in y direction
# rObs = np.array([[2.5],[2.5]] #vector of radii of other vehicles or obstacles (assumed to be circular) 
# zTraj #matrix containing the predicted trajectories of obstacles or other agents
# time = 15. #number of seconds spent in simulation
# freq = 10. #number of times to resolve problem during one second of simulated time

# #CAR DIMENSIONS
# roboRadius = 2.2 #longitudinal distance from center of control to furthest end of car body used as obstacle avoidance radius
# ELL = 2.65 #vehicle wheelbase; longitudinal distance between drive axes and directional wheel

# #CONTROL LIMITS
# speed_limit = 20. #max speed of vehicle in meters per second
# kappa_limit = np.tan(0.4) / ELL #maximum curvature of vehicle 