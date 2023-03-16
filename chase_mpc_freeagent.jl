################################################################################
# Intro Example
################################################################################
using Algames
using StaticArrays
using LinearAlgebra

include("driveSolverJulia.jl")

T = Float64

k = 10 #number of iterations in MPC
steps = 5 #time steps before replanning

# Define the dynamics of the system
p = 2 # Number of players
model = BicycleGame(p=p) # game with 3 players with unicycle dynamics
n = model.n
m = model.m

# Define the horizon of the problem
N = 20 # N time steps per solve
dt = 0.1 # each step lasts 0.1 second
probsize = ProblemSize(N,model) # Structure holding the relevant sizes of the problem

# Define the objective of each player
# We use a LQR cost
Q = [Diagonal([10,10,10,0].*ones(SVector{model.ni[i],T})) for i=1:p] # Quadratic state cost
R = [Diagonal(0.1*ones(SVector{model.mi[i],T})) for i=1:p] # Quadratic control cost

# Define the constraints that each player must respect
game_con = GameConstraintValues(probsize)
# Add collision avoidance
radius = 0.08
add_collision_avoidance!(game_con, radius)
# Add control bounds
u_max =  5*ones(SVector{m,T})
u_min = -5*ones(SVector{m,T})
add_control_bound!(game_con, u_max, u_min)
# # Add state bounds for player 1
x_max =  5*ones(SVector{n,T})
x_min = -5*ones(SVector{n,T})
add_state_bound!(game_con, 1, x_max, x_min)
# Add wall constraint
walls = [Wall([0.0,-0.4], [1.0,-0.4], [0.,-1.])]
add_wall_constraint!(game_con, walls)
# Add circle constraint
xc = [1., 2., 3.]
yc = [1., 2., 3.]
radius = [0.1, 0.2, 0.3]
add_circle_constraint!(game_con, xc, yc, radius)

# Designed to hold the trajectory of the entire rollout
trajTotal = zeros((steps*k,4,p+1));

# trajTotal[1,:,1] = start_1;
# trajTotal[1,:,2] = start_2;
# trajTotal[1,:,3] = start_3;

# print(trajTotal)

# Arrays defining initial state for each agent
start_1 = [0.1,-0.4,0.0,0.0];
# start_1 = [0.5,0.0,0.0,1.4];
start_2 = [0.0,0.0,0.0,0.0];
start_3 = [0.5;0.7;0.0;0.0];

# Column vectors defining desired state for each agent
r1 = radius[1] + radius[3] + 0.1; # minimum permissible distance between 1 and 3
goal_1 = [start_3[1]+r1*cos(start_3[4]),start_3[2]+r1*sin(start_3[4]),0,0];
# goal_1 = [2, 0.4,0,0];
r2 = radius[2] + radius[3] + 0.1;
goal_2 = [start_3[1]-r2*cos(start_3[4]),start_3[2]-r2*sin(start_3[4]),0,0];
# goal_2 = [2, 0.0,0,0];
goal_3 = [3;-0.4;0;0];

print(goal_1,"\n")
print(goal_2,"\n")
print(goal_3,"\n")

for iter in 1:k

    # Assuming opposing agents are stationary
    obsTraj = cat(repeat(start_1[1:2],1,N),repeat(start_2[1:2],1,N),dims=3)
    rTotal = 0.3*ones(N,p) #assuming dimensions of other players are constant

    # Plan the trajectory of the agent unaware of other agents' movements
    mpcTraj = trajPlan(start_3,goal_3,x_min[1],x_max[1],x_min[2],x_max[2],[-5,-5],[5,5],obsTraj,rTotal,N,0.3)
    global trajTotal[steps*(iter-1)+1:steps*iter,:,p+1] = mpcTraj[:,1:steps]'; #record relevant steps

    # Desrired state
    xf = [SVector{model.ni[1],T}(goal_1),
          SVector{model.ni[2],T}(goal_2),
          ]
    # Desired control
    uf = [zeros(SVector{model.mi[i],T}) for i=1:p]

    # Objectives of the game
    game_obj = GameObjective(Q,R,xf,uf,N,model)
    radius = 1.0*ones(p)
    μ = 5.0*ones(p)
    add_collision_cost!(game_obj, radius, μ)

    # Define the initial state of the system
    x0 = SVector{model.n,T}(reshape([
        start_1 start_2
        ]',(n,1)))

    # print([
    #     start_1 start_2 start_3
    #     ]',reshape([
    #         start_1 start_2 start_3
    #         ]',(n,1)))
    #

    # x0 = SVector{model.n,T}([
    #     0.1, 0.0, 0.5,
    #    -0.4, 0.0, 0.7,
    #     0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0,
    #     ])

    # Define the Options of the solver
    opts = Options()
    # Define the game problem
    prob = GameProblem(N,dt,x0,model,opts,game_obj,game_con)

    # Solve the problem
    @time newton_solve!(prob)

    # Visualize the Results
    # using Plots
    # plot(prob.model, prob.pdtraj.pr)
    # savefig("probmodel_pdtraj.png")

    # plot(prob.stats)
    # savefig("prob_stats.png")

    # print(size(prob.pdtraj.pr))

    trajXYVphi = zeros(N,4,p); # tensor containing subset of the trajectories calculated by the solver

    for i in 1:N
        # print("Step ",i,": ",prob.pdtraj.pr[i].z,"\n")
        tempTrajMat = reshape(prob.pdtraj.pr[i].z,(p,6));
        print("Step :",i,": ",tempTrajMat,"\n")
        for j in 1:p
            trajXYVphi[i,:,j] = tempTrajMat[j,1:4];
        end
    end
    # print("here")
    global trajTotal[steps*(iter-1)+1:steps*iter,:,1:p] = trajXYVphi[1:steps,:,:];
    # print("aqui")

    if iter != k
        global start_1 = trajXYVphi[steps+1,:,1];
        global start_2 = trajXYVphi[steps+1,:,2];
        global start_3 = mpcTraj[:,steps+1];

        idx = steps; #predicted location of pursued agent at end of next iteration

        # Calculate next goal positions of pursuing agents
        # global goal_1 = [trajXYVphi[idx,1,3]+r1*cos(trajXYVphi[idx,4,3]),trajXYVphi[idx,2,3]+r1*sin(trajXYVphi[idx,4,3]),0,0];
        global goal_1 = [mpcTraj[1,idx] + r1*cos(mpcTraj[4,idx]),mpcTraj[2,idx]+r1*sin(mpcTraj[4,idx]),0,0];
        # global goal_2 = [trajXYVphi[idx,1,3]-r2*cos(trajXYVphi[idx,4,3]),trajXYVphi[idx,2,3]-r2*sin(trajXYVphi[idx,4,3]),0,0];
        global goal_2 = [mpcTraj[1,idx] + r2*cos(mpcTraj[4,idx]),mpcTraj[2,idx]+r2*sin(mpcTraj[4,idx]),0,0];
    else
        # Calculate final goal positions of pursuing agents
        # global goal_1 = [trajXYVphi[steps,1,3]+r1*cos(trajXYVphi[steps,4,3]),trajXYVphi[steps,2,3]+r1*sin(trajXYVphi[steps,4,3]),0,0];
        # global goal_2 = [trajXYVphi[steps,1,3]-r2*cos(trajXYVphi[steps,4,3]),trajXYVphi[steps,2,3]-r2*sin(trajXYVphi[steps,4,3]),0,0];
        global goal_1 = [mpcTraj[1,steps] + r1*cos(mpcTraj[4,steps]),mpcTraj[2,steps]+r1*sin(mpcTraj[4,steps]),0,0];
        global goal_2 = [mpcTraj[1,steps] + r2*cos(mpcTraj[4,steps]),mpcTraj[2,steps]+r2*sin(mpcTraj[4,steps]),0,0];
    end
    # print("hier")
end

using Plots
# print(size(trajTotal))
# print(trajTotal)
plot(trajTotal[:,1,:],trajTotal[:,2,:])
savefig("trajectory_total.png")

for player in 1:p
    for paso in 1:steps*k
        print("\n",trajTotal[paso,:,player])
    end
    print("\n")
end