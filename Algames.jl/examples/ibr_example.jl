################################################################################
# Iterative Best Response Example
################################################################################
# using Algames
using StaticArrays
using LinearAlgebra


T = Float64

# Define the dynamics of the system
p = 3 # Number of players
d = 2
model = DoubleIntegratorGame(p=p, d=d) # game with 3 players with unicycle dynamics
n = model.n
m = model.m

# Define the horizon of the problem
N = 20 # N time steps
dt = 0.1 # each step lasts 0.1 second
probsize = ProblemSize(N,model) # Structure holding the relevant sizes of the problem

# Define the objective of each player
# We use a LQR cost
Q = [Diagonal(50*ones(SVector{model.ni[i],T})) for i=1:p] # Quadratic state cost
R = [Diagonal(0.01*ones(SVector{model.mi[i],T})) for i=1:p] # Quadratic control cost
# Desrired state
xf = [SVector{model.ni[1],T}([ 0, 0*0.0,0,0]),
      SVector{model.ni[1],T}([ 0, 0*0.2,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*0.4,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*0.6,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*0.8,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*1.0,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*1.2,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*1.4,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*1.6,0,0]),
	  SVector{model.ni[1],T}([ 0, 0*1.8,0,0]),
      ]
xf = xf[1:p]
# Desired control
uf = [zeros(SVector{model.mi[i],T}) for i=1:p]
# Objectives of the game
game_obj = GameObjective(Q,R,xf,uf,N,model)
radius = 3.0*ones(p)
μ = 2.0*ones(p)
add_collision_cost!(game_obj, radius, μ)

# Define the constraints that each player must respect
game_con = GameConstraintValues(probsize)
# # Add collision avoidance
radius = 0.25
add_collision_avoidance!(game_con, radius)

# Define the initial state of the system
x0 = [
	[ 0.00,  0.0,  0.0,  0.0],
	[ 0.05,  0.2,  0.0,  0.0],
	[ 0.10,  0.4,  0.0,  0.0],
	[ 0.15,  0.6,  0.0,  0.0],
	[ 0.20,  0.8,  0.0,  0.0],
	[ 0.25,  1.0,  0.0,  0.0],
	[ 0.30,  1.2,  0.0,  0.0],
	[ 0.35,  1.4,  0.0,  0.0],
	[ 0.40,  1.6,  0.0,  0.0],
	[ 0.45,  1.8,  0.0,  0.0],
	]
θ = range(0.0, length=2, stop=2π)
rad = 0.5
x0 = [[(rad+θ/10)*cos(θ), (rad+θ/10)*sin(θ), 0, 0] for θ in range(0.0, length=p, stop=2π*(1-1/p))]
x0 = SVector{model.n,T}(reshape(vcat([x0[i]' for i=1:p]...), model.n))


# Define the Options of the solver
opts_alg = Options(Δ_min=1e-9, inner_print=false)
opts_ibr = Options(dual_reset=false, outer_iter=10, inner_iter=25, Δ_min=1e-9, inner_print=false)
# opts_ibr = Options(dual_reset=false, outer_iter=7, inner_iter=20, Δ_min=1e-5, inner_print=false)
# ibr_opts = IBROptions(Δ_min=1e-9, ibr_iter=100, ordering=shuffle!([1,2,3,4,5,6,7,8,9,10]))
ibr_opts = IBROptions(Δ_min=1e-9, ibr_iter=15, ordering=[1,2,3,4,5,6,7,8,9,10], live_plotting=true)
ibr_opts = IBROptions(Δ_min=1e-9, ibr_iter=100, ordering=[2,1,3,4,5,6,7,8,9,10], live_plotting=false)

# Define the game problem
prob0_alg = GameProblem(N,dt,x0,model,opts_alg,game_obj,game_con)
prob0_ibr = GameProblem(N,dt,x0,model,opts_ibr,game_obj,game_con)
prob_alg = deepcopy(prob0_alg)
prob_ibr = deepcopy(prob0_ibr)

# cost.(prob_ibr.game_obj.obj)
# game_obj.obj

# Solve the problem
@elapsed newton_solve!(prob_alg)
@elapsed ibr_newton_solve!(prob_ibr, ibr_opts=ibr_opts)

residual!(prob_ibr, prob_ibr.pdtraj)
norm(prob_ibr.core.res, 1)/length(prob_ibr.core.res)
residual!(prob_alg, prob_alg.pdtraj)
norm(prob_alg.core.res, 1)/length(prob_alg.core.res)
prob_ibr.stats

plt = plot()
plot!(plt, cumsum(prob_alg.stats.t_elap), log.(10, prob_alg.stats.res))
plot!(plt, cumsum(prob_alg.stats.t_elap), log.(10, [v.max for v in prob_alg.stats.dyn_vio]))
plot!(plt, cumsum(prob_alg.stats.t_elap), log.(10, [v.max for v in prob_alg.stats.sta_vio]))
plot!(plt, cumsum(prob_ibr.stats.t_elap), log.(10, prob_ibr.stats.res))
plot!(plt, cumsum(prob_ibr.stats.t_elap), log.(10, [v.max for v in prob_ibr.stats.dyn_vio]))
plot!(plt, cumsum(prob_ibr.stats.t_elap), log.(10, [v.max for v in prob_ibr.stats.sta_vio]))

# Visualize the Results
Algames.plot_traj!(prob_alg.model, prob_alg.pdtraj.pr)
Algames.plot_traj!(prob_ibr.model, prob_ibr.pdtraj.pr)
Algames.plot_violation!(prob_alg.stats)
Algames.plot_violation!(prob_ibr.stats)
const Algames = Main
using Plots

# IBR
[prob_ibr.game_con.state_conval[1][1].λ[k][1] for k=1:N-1][end]
[prob_ibr.game_con.state_conval[1][2].λ[k][1] for k=1:N-1][end]

[prob_ibr.game_con.state_conval[2][1].λ[k][1] for k=1:N-1][end]
[prob_ibr.game_con.state_conval[2][2].λ[k][1] for k=1:N-1][end]

[prob_ibr.game_con.state_conval[3][1].λ[k][1] for k=1:N-1][end]
[prob_ibr.game_con.state_conval[3][2].λ[k][1] for k=1:N-1][end]

# Algames
[prob_alg.game_con.state_conval[1][1].λ[k][1] for k=1:N-1][end]
[prob_alg.game_con.state_conval[1][2].λ[k][1] for k=1:N-1][end]

[prob_alg.game_con.state_conval[2][1].λ[k][1] for k=1:N-1][end]
[prob_alg.game_con.state_conval[2][2].λ[k][1] for k=1:N-1][end]

[prob_alg.game_con.state_conval[3][1].λ[k][1] for k=1:N-1][end]
[prob_alg.game_con.state_conval[3][2].λ[k][1] for k=1:N-1][end]


#Conclusions
	# - if there is no coupling at all (dynamics, cost, state constraints)
	# IBR marginally better, but should be the same since we exploit sparsity using a sparse linear solver.

	# - if there is cost coupling only
	# IBR converges to a different local NE
	# the solution depends on the players' ordering
	# the solution does not reflect the symmetry of information between players
	# some players having cost large advantages over

	# - if there is state constraints coupling (when state constraint are active)
	# IBR converges to a different local NNE
	# the solution depends on the players' ordering
	# the solution does not reflect the symmetry of information between players
	# some players having large cost advantages over
	# converges to a solution which is also a Stackelberg equilibrium for 2 players,
	# and chain-Stackelberg for multi player scenarios.
	# really hard time converging the solutions are oscillating in practice
