using LinearAlgebra
using Convex
using ECOS

function initializer(xInitial,xFinal,minX,maxX,minY,maxY,obsTraj,rTotal,N,agentRadius,conservative=false)
    
    print("Initializing trajectory solver\n")

    if conservative
        return repeat(xInitial,1,N) #constant repetitions of original value
    end
    
    n = length(xInitial)
    m = 2    
    obstacles = size(rTotal)[2]; #total number of obstacles (incl. other agents) to dodge
    
    xt = hcat(LinRange(xInitial,xFinal,N)...); # vector of vectors with initial guess of locations
    
    # Loop Control
    diff = Inf; #initialize to unreasonable value to overwrite in loop
    epsilon = 1e-3; #tolerance for convergence of solution
    
    optval = Inf; # objective function value
    iter = 0; # number of iterations of convex solver
    
    x = Variable(n,N); #state vectors arranged as matrix
    
    while abs(diff) > epsilon && iter <= 150
        # Initial and final position constraints
        constraints = [x[:,1] == xInitial]; #initialize constraints list with initial position
        constraints = cat(constraints,[x[:,N] == xFinal],dims=1); #hard constraint on final position
        
        # Work space constraints
        constraints = cat(constraints,[x[2,:] >= minY + agentRadius, x[2,:] <= maxY - agentRadius],dims=1) # Y axis
        constraints = cat(constraints,[x[1,:] >= minX + agentRadius, x[1,:] <= maxX - agentRadius],dims=1) # X axis
        
        # Obstacle avoidance constraints
        zt = xt[1:2,:]; #isolated positional values, 2xN matrix
        
        zSub = repeat(zt,1,1,obstacles) - obsTraj #2xNxO tensor containing diffferences between each position in the trajectory and position of each obstacle center
        xSub = zSub[1,:,:] #NxO matrix
        ySub = zSub[2,:,:] #NxO matrix
        
        zDiffs = x[1:2,:] - zt #2xN matrix, difference between current and prior position trajectory
        xDiffs = zDiffs[1,:] # 1xN vector
        yDiffs = zDiffs[2,:] # 1xN vector
        
        zSubNorms = sqrt.(sum(zSub.^2,dims=1))[1,:,:] #NxO matrix of Euclidean distances between each position in the trajectory and each obstacle center
        
        for o in 1:obstacles
            constraints = cat(constraints,[rTotal[:,o] - zSubNorms[:,o] - (xSub[:,o].* xDiffs' + ySub[:,o].* yDiffs')./zSubNorms[:,o] <= 0],dims=1) #o-th obstacle avoidance constraint
        end
        
        problem = minimize(sumsquares(x[:,2:N] - x[:,1:N-1]), constraints)
            
        solve!(problem, ECOS.Optimizer)
        
        mincost = problem.optval;
        
        print("initializer opt :",mincost,"\n") #monitor
        print("initializer problem status: ",problem.status,"\n")

        diff = maximum(abs.(x.value - xt)) #for checking convergence of trajectory solution
        
        if problem.status == "INFEASIBLE" #if bad information from the environment provokes an infeasible solve, don't crash
            xt = initializer(xInitial,xFinal,minX,maxX,minY,maxY,obsTraj,rTotal,N,agentRadius,conservative=true)
        else
            xt = copy(x.value); #for use in following iteration
        end

        zt = x[1:2,:] #ACTIVATE FOR OBSTACLE AVOIDANCE CONSTRAINTS
        print("initializer convergence measure: ",diff,"\n") #monitor
        iter += 1;

    end

#     print("initializer optimal value: ",mincost)

    return xt
    
end

function trajPlan(xInitial,xFinal,minX,maxX,minY,maxY,minU,maxU,obsTraj,rTotal,N,agentRadius)
    
    n = length(xInitial)
    m = 2
    obstacles = size(rTotal)[2]; #total number of obstacles (incl. other agents) to dodge
    
    xt = initializer(xInitial,xFinal,minX,maxX,minY,maxY,obsTraj,rTotal,N,agentRadius); # vector of vectors with initial guess of locations
    
    # Loop Control
    diff = Inf; #initialize to unreasonable value to overwrite in loop
    epsilon = 1e-3; #tolerance for convergence of solution
    
    optval = Inf; # objective function value
    iter = 0; # number of iterations of convex solver
    
    x = Variable(n,N); #state vectors arranged as matrix
    u = Variable(m,N); #action vectors arranged as matrix
    dt = 0.1; #time difference between time steps
    
    
    while abs(diff) > epsilon && iter <= 150

        # Original objective function
        objective = norm((x[:,N] - xFinal).*[10;10;1;1],1) + dt*square(norm(vec(u)));
        
        # Initial and final position constraints
        constraints = [x[:,1] == xInitial]; #initialize constraints list with initial position
#         constraints = cat(constraints,[x[:,N] == xFinal],dims=1); #hard constraint on final position
        
        # Work space constraints
        constraints = cat(constraints,[x[2,:] >= minY + agentRadius, x[2,:] <= maxY - agentRadius],dims=1) # Y axis
        constraints = cat(constraints,[x[1,:] >= minX + agentRadius, x[1,:] <= maxX - agentRadius],dims=1) # X axis
                
        # Control inputs
        constraints = cat(constraints,[x[3,2:N] == x[3,1:N-1] + dt*u[1,1:N-1]],dims=1) #acceleration
        constraints = cat(constraints,[x[4,2:N] == x[4,1:N-1] + dt*u[2,1:N-1]],dims=1) #steering angle
        
        # Control limits
        constraints = cat(constraints,[u <= repeat(maxU,1,20)],dims=1) #max inputs
        constraints = cat(constraints,[u >= repeat(minU,1,20)],dims=1) #min inputs
        
        # Dynamics constraints
#         prev_psi = reshape(xt[4,:],1,:); #history of headings
#         prev_veloc = reshape(xt[3,:],1,:); #history of velocities      
        prev_psi = xt[4,:];
        prev_veloc = xt[3,:];
              
        for steppin in 1:N-1
            constraints = cat(constraints,[x[1,steppin+1] == x[1,steppin] + dt*(x[3,steppin]*cos(prev_psi[steppin]) - prev_veloc[steppin]*sin(prev_psi[steppin])*(x[4,steppin]-prev_psi[steppin]))],dims=1)
#             constraints = cat(constraints,[x[1,steppin+1] == x[1,steppin] + dt*(x[3,1]*cos(prev_psi[1,1]) - prev_veloc[1,1]*sin(prev_psi[1,1])*(x[4,1]-prev_psi[1,1]))],dims=1)
            constraints = cat(constraints,[x[2,steppin+1] == x[2,steppin] + dt*(x[3,steppin]*sin(prev_psi[steppin]) + prev_veloc[steppin]*cos(prev_psi[steppin])*(x[4,steppin]-prev_psi[steppin]))],dims=1) #y position change

        end
        
#         constraints = cat(constraints,[x[1,2:N] == x[1,1:N-1] + dt*(x[3,1:N-1].*reshape(cos.(prev_psi[:,1:N-1]),1,:) - prev_veloc[:,1:N-1].*reshape(sin.(prev_psi[:,1:N-1]),1,:).*(x[4,1:N-1]-prev_psi[:,1:N-1]))],dims=1) #x position change
#         constraints = cat(constraints,[x[2,2:N] == x[2,1:N-1] + dt*x[3,1:N-1].*sin.(prev_psi[:,1:N-1]) + dt*prev_veloc[:,1:N-1].*cos.(prev_psi[:,1:N-1]).*(x[4,1:N-1]-prev_psi[:,1:N-1])],dims=1) #y position change
        
        # Obstacle avoidance constraints
        zt = xt[1:2,:]; #isolated positional values, 2xN matrix
        
        zSub = repeat(zt,1,1,obstacles) - obsTraj #2xNxO tensor containing diffferences between each position in the trajectory and position of each obstacle center
        xSub = zSub[1,:,:] #NxO matrix
        ySub = zSub[2,:,:] #NxO matrix
        
        zDiffs = x[1:2,:] - zt #2xN matrix, difference between current and prior position trajectory
        xDiffs = zDiffs[1,:] # 1xN vector
        yDiffs = zDiffs[2,:] # 1xN vector
        
        zSubNorms = sqrt.(sum(zSub.^2,dims=1))[1,:,:] #NxO matrix of Euclidean distances between each position in the trajectory and each obstacle center
        
        for o in 1:obstacles
#             constraints = cat(constraints,[rTotal[:,o] - zSubNorms[:,o] - (xSub[:,o].* xDiffs' + ySub[:,o].* yDiffs')./zSubNorms[:,o] <= 0],dims=1) #o-th obstacle avoidance constraint
            objective += 10*sum(max(0.1 .+ rTotal[:,o] - zSubNorms[:,o] - (xSub[:,o].* xDiffs' + ySub[:,o].* yDiffs')./zSubNorms[:,o],0)) #soft version of obstacle avoidance constraints
        end
        
        problem = minimize(objective, constraints)
            
        solve!(problem, ECOS.Optimizer)
        
        mincost = problem.optval;
        
        print("Trajectory solver opt :",mincost,"\n") #monitor
        print("Trajectory solver problem status: ",problem.status,"\n")

        diff = maximum(abs.(x.value - xt)) #for checking convergence of trajectory solution
        
        if problem.status == "INFEASIBLE" #if bad information from the environment provokes an infeasible solve, don't crash
            xt = initializer(xInitial,xFinal,minX,maxX,minY,maxY,obsTraj,rTotal,N,agentRadius,conservative=true)
        else
            xt = copy(x.value); #for use in following iteration
#             ut = copy(u.value); #for output
        end

        zt = x[1:2,:] #ACTIVATE FOR OBSTACLE AVOIDANCE CONSTRAINTS
        print("Trajectory solver convergence measure: ",diff,"\n") #monitor
        iter += 1;

    end

    return xt #,ut
end