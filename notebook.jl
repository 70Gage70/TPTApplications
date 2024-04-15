### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b2478af7-9c04-4f01-b261-b2db30c52a5c
begin
	using PlutoUI, HypertextLiteral
	using POMDPs    
	using POMDPModelTools
	using POMDPSimulators
	using POMDPPolicies
	using QuickPOMDPs
	using DiscreteValueIteration
	using Parameters
	using Random
	using ColorSchemes, Colors
	using Latexify
	using Plots
	using TransitionPathTheory
	default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	@info "Loaded dependencies."
end

# ╔═╡ 53fc4147-2716-4b36-9752-ffc3693d1c72
md"""
# Transition Path Theory In Grid World
"""

# ╔═╡ e2377ddd-03fe-424d-82e9-8d90d2c5ac56
md"""
---
---
"""

# ╔═╡ 56cd93e8-5794-4dfa-aa66-28a550ad7849
md"""
``(x, y)`` = $(@bind state_xy confirm(TextField(default = "(8, 3)"), label = "SET"))
"""

# ╔═╡ 412f6c4c-2209-4542-9de9-d529ca5bd087
md"""
---
---
"""

# ╔═╡ a8817d6e-2302-4f39-8b93-66d550ca09ef
md"""
### N iterations $(@bind vi_iterations Slider(0:30, default=0, show_value=true))
"""

# ╔═╡ 3590b4b0-5a05-4052-aa58-f48d3912ce77
md"""
### Discount $(@bind γ_vi Slider(0:0.05:1, default=0.95, show_value=true))
"""

# ╔═╡ 6864700f-4705-4c8c-97c8-24d119dbcbbc
md"""
---
---
"""

# ╔═╡ d5df4c2e-ab77-4331-8f6d-4fd76c44bbb9
md"""
New simulation: $(@bind recompute_simul Button())
"""

# ╔═╡ 34326f2e-b804-481a-b151-0f3529ac8b3a
md"""
---
---
"""

# ╔═╡ 7aa7700b-490e-4929-9c78-cca7840ff86b
md"""
## Transition Path Theory
"""

# ╔═╡ 2adbfa59-3d25-4819-b1f6-75b7979462c1
md"""
### Diagnostic Rewards
(9, 3, 50, 0)
(4, 3, -10, 1)
(4, 6, -5, 0)

(2, 9)

### Cool Rewards
(1, 11, 3, 1)
(10, 8, -3, 0)
(10, 9, -3, 0)
(10, 10, -3, 0)
(10, 11, -3, 0)
(11, 11, 6, 1)

(6, 1)
"""

# ╔═╡ 5e9b28a4-50b9-4c44-8d3c-48eb72bda3a5
md"""
## Grid World Problem (MDP)
In the _Grid World_ problem, an _agent_ moves around a grid attempting to collect as much reward as possible, trying to avoid negative rewards.
"""

# ╔═╡ becac04c-9819-42b7-811e-034440796511
md"""
### Helpers
"""

# ╔═╡ cd693f7d-1576-4c2b-a60b-67b45a03ecf2
"""
	NothingPolicy
"""
struct NothingPolicy <: Policy end

# ╔═╡ 3cf71e78-d2b9-4267-9593-58dd46100033
"""
	one_based_policy!(policy)
"""
function one_based_policy!(policy)
	# change the default action in the policy (all zeros) to all ones (if needed)
	if all(iszero, policy.policy)
		policy.policy[:] = ones(eltype(policy.policy), length(policy.policy))
	end
end

# ╔═╡ e2a84ebf-a259-43c1-b512-f6c6b6e02d14
md"""
### Setup
"""

# ╔═╡ 5e5dbd4b-85f5-466a-9763-7dc936887477
struct State
	x::Int
	y::Int
end

# ╔═╡ c1d07fca-1fbd-4450-96b1-c829d7ad8306
Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)

# ╔═╡ c092511d-c2e7-4b8c-8104-b4b10893cb02
@with_kw struct GridWorldParameters
	size::Tuple{Int,Int} = (11, 11)   # size of the grid
	null_state::State = State(-1, -1) # terminal state outside of the grid
	p_transition::Real = 0.7 # probability of transitioning to the correct next state
end	

# ╔═╡ 13dbf845-14a7-4c98-a1db-b3a83c9ce37c
params = GridWorldParameters();

# ╔═╡ 4a14aee4-12f1-4d55-9532-9b88e4c465f8
𝒮 = [[State(x,y) for x=1:params.size[1], y=1:params.size[2]]..., params.null_state]

# ╔═╡ a90137f2-b058-4e50-a3cf-9cae179c774d
"""
	state2idx
"""
state2idx = Dict(𝒮[i] => i for i = 1:length(𝒮))

# ╔═╡ bcc5e8a3-1e3a-40cf-a306-13599a4952ac
@enum Action UP DOWN LEFT RIGHT

# ╔═╡ bc541507-61db-4084-9712-1c57d139e17f
𝒜 = [UP, DOWN, LEFT, RIGHT]

# ╔═╡ 1303be2a-d18c-44b0-afb9-06a6b4ce5c08
begin
	const MOVEMENTS = Dict(UP    => State(0,1),
						   DOWN  => State(0,-1),
						   LEFT  => State(-1,0),
						   RIGHT => State(1,0));

	Base.:+(s1::State, s2::State) = State(s1.x + s2.x, s1.y + s2.y)
end

# ╔═╡ 1f55df07-0f67-4ed4-b5f5-5a46fb28ab85
"""
	plot_transition_probability(distr)
"""
function plot_transition_probability(distr)
	if distr isa Deterministic
		vals = [distr.val]
		probs = [1]
	else
		vals = distr.vals
		probs = distr.probs
	end
	xtick_states = [(v.x,v.y) for v in vals]
	@info xtick_states
	@info probs
	cmap_probs = ColorScheme([colorant"#B3BCDB", colorant"#4063D8"])
	bar(probs,
		xformatter = x->1 <= x <= length(xtick_states) ? string(xtick_states[Int(x)]) : "",
		label=false,
		aspect_ratio=5,
		group=xtick_states,
		color=map(v->get(cmap_probs, v), probs))
	title!("Transition Probability\nDistribution")
	xlabel!("next state 𝑠′")
	ylabel!("probability")
	ylims!(0, 1.1)
	xlims!(0, 6)
	yticks!(0:0.1:1)
end

# ╔═╡ 49901c66-db64-48a2-b122-84d5f6b769db
"""
	inbounds(s)
"""
inbounds(s::State) = 1 ≤ s.x ≤ params.size[1] && 1 ≤ s.y ≤ params.size[2]

# ╔═╡ 28efe3a2-36bd-48d5-9762-7410c7498856
"""
	tuplestring2vec(s)
"""
tuplestring2vec(s::AbstractString) = parse.(Int64, split(replace(s, "(" => "", ")" => ""), ","))

# ╔═╡ 87a6c45e-6f3e-428e-8301-3b0c4166a84b
md"""
### Discount: $(@bind γ Slider(0:0.05:1, default=0.95, show_value=true))
"""

# ╔═╡ 6970821b-2b87-4e66-b737-512e83627998
termination(s::State) = s == params.null_state

# ╔═╡ 49b140ad-641f-436c-9492-cf3efbadd8d2
abstract type GridWorld <: MDP{State, Action} end

# ╔═╡ 59f9c300-25ba-4193-991b-7da9df00d78d
"""
	get_rewards(mdp, policy)
"""
function get_rewards(mdp::QuickMDP{GridWorld}, policy::Policy)
	null_state = params.null_state
	valid_states = setdiff(states(mdp), [null_state])
	U = map(s->reward(mdp, s), valid_states)
end

# ╔═╡ 108efbd5-5ff8-4d17-aab4-15702dd4ef2d
"""
	values(mdp, policy)

`policy` may be a `Policy`, a `ValuePolicy` or `Union{NothingPolicy, FunctionPolicy}`
"""
function values(mdp::QuickMDP{GridWorld}, policy::Policy)
	null_state = params.null_state
	valid_states = setdiff(states(mdp), [null_state])
	U = map(s->value(policy, s), valid_states)
end

# ╔═╡ b307378e-d5eb-46a6-99b5-7032d9926c8e
function values(mdp::QuickMDP{GridWorld}, policy::ValuePolicy)
	maxU = mapslices(maximum, policy.value_table, dims=2)
	return maxU[1:end-1] # remove null_state
end

# ╔═╡ bf240f83-4fc9-4fe2-8936-b0278500204d
function values(mdp::QuickMDP{GridWorld}, policy::Union{NothingPolicy, FunctionPolicy})
	null_state = params.null_state
	valid_states = setdiff(states(mdp), [null_state])
	rewards = map(s->reward(mdp, s), valid_states)
end

# ╔═╡ e67b6b1e-0cb3-4840-b566-8ae9d3d3cdad
md"""
### Solving
"""

# ╔═╡ 48c966cb-6b79-42a6-8ff0-2fe3261f3981
solver = ValueIterationSolver(max_iterations=30);

# ╔═╡ a4e1b417-60c9-475e-aa16-b2112f7b47cf
md"""
### Extra
"""

# ╔═╡ 87710f14-a7ec-4983-b105-8090c4dd1463
cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")

# ╔═╡ 92cdb96c-e636-4308-bf49-6d34246d3256
ColorScheme([get(cmap, i) for i in 0.0:0.001:1.0])

# ╔═╡ 9c139d5b-b8c9-4e00-ba5b-ee66fad73658
begin
	eureka =[
	    256 256 255;
	    153 204 255;
	    255 255 102;
	    204 102   0;
	    102   0   0]/256
	
	EUREKA = [RGB{Float64}(eureka[i,:]...) for i = 1:size(eureka, 1)] |> ColorScheme
end

# ╔═╡ aeec3109-1ebc-4edf-9ac5-eda3a30467cc
md"""
---
"""

# ╔═╡ 99f2318b-b1d2-4afa-960b-eed2971abede
begin
	@info "Setting notebook width."
	html"""
	<style>
		main {
			margin: 0 auto;
			max-width: 2000px;
	    	padding-left: 10%;
	    	padding-right: 30%;
		}
	</style>
	"""
end

# ╔═╡ b51e4f52-a3ad-479c-b4de-59261389e409
begin

@info "Hiding header"

html"""
<style>
    header {
		display:none
    }
</style>
"""
end

# ╔═╡ 576bc4b1-3528-47b9-9345-3a5058327c28
ad(text, kind) = Markdown.MD(Markdown.Admonition(kind, "", [text]))

# ╔═╡ acfe17fe-0467-4f0f-8fcb-bc5e6b6f01be
begin
	@info "Defining reward setting interface."

	local blurb = md"""
	- Entries of the form ``(x, y, \text{value}, T)``
	- separated by new lines
	- ``T`` is one of 
	  - 1 (TPT target, ``B``)
	  - 0 (TPT avoid)
	- ``A`` is the TPT source state
	"""
	
	local reward_params_box = md"""
		$(details("HELP", blurb))
		$(@bind reward_params2 confirm(TextField((18, 3), default= "(9, 3, 50, 1)\n(4, 3, -10, 0)\n(4, 6, -5, 0)" ), label = "SET"))
		"""
	local tpt_box = md"""
	``A`` = $(@bind tpt_A confirm(TextField(3, default = "(2, 9)"), label = "SET"))
	"""

	local reward_params_box2 = ad(
		details("REWARDS+TPT", [reward_params_box, tpt_box], open = true), 
	"info")

	@htl """<div style="
	position: fixed; 
	right: 1rem; 
	top: 8rem; 
	padding: 1px;
	text-align: left;
	z-index: 98;
	max-width: 24%;
	max-height: 20%;
	background-color: var(--main-bg-color);">
	$(reward_params_box2)
	</div>"""
end

# ╔═╡ f7814a66-23c8-4782-ba06-755397af87db
"""
	R(s, a)
"""
function R(s, a=missing)
	rewards = tuplestring2vec.(split(reward_params2, "\n"))

	for r in rewards
		if s.x == r[1] && s.y == r[2]
			return r[3]
		end
	end

	return 0
end

# ╔═╡ 52b96024-9f52-4f07-926b-2297ed7dd166
"""
	policy_grid(policy, xmax, ymax)
"""
function policy_grid(policy::Policy, xmax::Int, ymax::Int)
    arrows = Dict(UP => "↑",
                  DOWN => "↓",
                  LEFT => "←",
                  RIGHT => "→")

    grid = Array{String}(undef, xmax, ymax)
    for x = 1:xmax, y = 1:xmax
        s = State(x, y)
        grid[x,y] = R(s) == 0 ? arrows[action(policy, s)] : ""
    end

    return grid
end

# ╔═╡ 25d170e9-1c26-4058-96fc-04ab47964b51
"""
	plot_grid_world(mdp, policy = NothingPolicy(), iter = 0, discount = NaN; outline, show_policy, extra_title, show_rewards, rewards_color, outline_state, force_blank, plot_vector)
"""
function plot_grid_world(mdp::MDP,
        policy::Policy=NothingPolicy(),
        iter=0,
        discount=NaN;
        outline=true,
        show_policy=true,
        extra_title=isnan(discount) ? "" : " (iter=$iter, γ=$discount)",
        show_rewards=false,
		rewards_color = :white,
        outline_state::Union{State, Nothing}=nothing,
		outline_A::Union{Vector{<:State}, Nothing}=nothing,
		outline_B::Union{Vector{<:State}, Nothing}=nothing,
		outline_avoids::Union{Vector{<:State}, Nothing}=nothing,
		force_blank=false,
		plot_vector::Union{Vector, Nothing}=nothing,
		plot_arrows::Union{Vector{<:Vector}, Nothing}=nothing)
    
    gr()
	
    if policy isa NothingPolicy
        # override when the policy is empty
        show_policy = false
    end
    
    if iter == 0
        # solver has not been run yet, so we just plot the raw rewards
        # overwrite policy at time=0 to be emp
        U = get_rewards(mdp, policy)
    else
        # otherwise, use the Value Function to get the values (i.e., utility)
        U = values(mdp, policy)
    end

    # reshape to grid
    (xmax, ymax) = params.size
	
    Uxy = reshape(U, xmax, ymax)

	rewards = tuplestring2vec.(split(reward_params2, "\n"))

	if force_blank
		Uxy = fill(NaN, xmax, ymax)

		for r in rewards
			Uxy[r[1], r[2]] = r[3] != 0 ? r[3] : NaN
		end
	end


    # plot values (i.e the U matrix)
	heat = plot_vector === nothing ? Uxy' : reshape(plot_vector, xmax, ymax)'
    fig = heatmap(heat,
                  legend=:none,
                  aspect_ratio=:equal,
                  framestyle=:box,
                  tickdirection=:out,
				  grid = false,
                  color=cmap.colors)
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    
    for x in 1:xmax, y in 1:ymax
        # display policy on the plot as arrows
        if show_policy
            grid = policy_grid(policy, xmax, ymax)
            annotate!([(x, y, (grid[x,y], :center, 12, "Computer Modern"))])
        end
        if outline
            rect = rectangle(1, 1, x - 0.5, y - 0.5)
            plot!(rect, fillalpha=0, linecolor=:gray)
        end
    end

    if !isnothing(outline_state)
        terminal_states = filter(s->reward(mdp, s) != 0, states(mdp))
        color = (outline_state in terminal_states) ? "yellow" : "blue"
        rect = rectangle(1, 1, outline_state.x - 0.5, outline_state.y - 0.5)
        plot!(rect, fillalpha=0, linecolor=color, linewidth = 4)
    end

	if !isnothing(outline_A)
		for s_A in outline_A
	        rect = rectangle(1, 1, s_A.x - 0.5, s_A.y - 0.5)
	        plot!(rect, fillalpha=0, linecolor=:blue, linewidth = 8)
		end
	end

	if !isnothing(outline_B)
		for s_B in outline_B
	        rect = rectangle(1, 1, s_B.x - 0.5, s_B.y - 0.5)
	        plot!(rect, fillalpha=0, linecolor=:orange, linewidth = 8)
		end
	end

	if !isnothing(outline_avoids)
		for s_av in outline_avoids
	        rect = rectangle(1, 1, s_av.x - 0.5, s_av.y - 0.5)
	        plot!(rect, fillalpha=0, linecolor=:gray, linewidth = 8)
		end
	end

    if show_rewards
        for s in filter(s->reward(mdp, s) != 0, states(mdp))
            r = reward(mdp, s)
            annotate!([(s.x, s.y, (r, rewards_color, :center, 12, "Computer Modern"))])
        end
    end

	if !isnothing(plot_arrows)
		fi = plot_arrows
		mags = [(f[1]^2 + f[2]^2)^(1/8) for f in fi]
		magsmin = filter(x -> x > 0, mags) |> minimum
		magsmax = maximum(mags)
		mags = map(x -> x == 0.0 ? 0.0 : (x - magsmin)/(magsmax - magsmin), mags)
		dirs = [mags[i] == 0.0 ? [0.0, 0.0] : fi[i]/sqrt(fi[i][1]^2 + fi[i][2]^2) for i = 1:length(fi)] .* 0.4
		xy = Iterators.product(1:params.size[1], 1:params.size[2]) |> collect
		u = [dirs[i][1] for i = 1:length(dirs)]
		v = [dirs[i][2] for i = 1:length(dirs)]
			
		for i = 1:length(dirs)
			quiver!([xy[i][1]], [xy[i][2]], quiver=([u[i]], [v[i]]), 
				color = get(EUREKA, mags[i]),
				linewidth = 3)
		end
	end
	
    title!("$extra_title")

    return fig
end

# ╔═╡ c85a316a-acd3-4aa3-8b65-28a332950240
render = plot_grid_world

# ╔═╡ 27e554ff-9861-4a41-ad65-9d5ae7727e45
"""
	T(s, a)
"""
function T(s::State, a::Action)
	if R(s) != 0
		return Deterministic(params.null_state)
	end

	Nₐ = length(𝒜)
	next_states = Vector{State}(undef, Nₐ + 1)
	probabilities = zeros(Nₐ + 1)
	p_transition = params.p_transition

	for (i, a′) in enumerate(𝒜)
		prob = (a′ == a) ? p_transition : (1 - p_transition) / (Nₐ - 1)
		destination = s + MOVEMENTS[a′]
		next_states[i+1] = destination

		if inbounds(destination)
			probabilities[i+1] += prob
		end
	end
	
	# handle out-of-bounds transitions
	next_states[1] = s
	probabilities[1] = 1 - sum(probabilities)

	return SparseCat(next_states, probabilities)
end

# ╔═╡ 5af462ea-781b-4509-98de-c68c29ec81fd
mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ,
    initialstate = 𝒮,
    isterminal   = termination,
	render       = render);

# ╔═╡ b6deb2b4-39eb-48e9-8bbe-340121c3de2e
render(mdp; show_rewards=true, extra_title = "Grid World Setup", outline = true, force_blank = true)

# ╔═╡ ba95fdd5-d643-4de2-8235-90e2b5651410
policy = solve(solver, mdp)

# ╔═╡ cd41b63d-3fc3-4fdd-871a-d0d0a3ee279d
begin
	state_x, state_y = tuplestring2vec(state_xy)
	sᵣ = State(state_x, state_y)
	aᵣ = action(policy, sᵣ)
	@info "Defining example state, action."
end

# ╔═╡ a4230ab4-a655-44f2-8292-e28af4e702cb
begin
	distr = transition(mdp, sᵣ, aᵣ)
	
	grid_plot = render(mdp, policy, 0; 
		outline=true, 
		outline_state=sᵣ, 
		show_policy = true,
		show_rewards = true,
		force_blank = true)
	distr_plot = plot_transition_probability(distr)
	plot(grid_plot, distr_plot, layout=2)
end

# ╔═╡ bc474e19-7d3e-481c-b36c-e231cd72db94
begin
	recompute_simul
	
	steps = collect(stepthrough(mdp, policy, "s,a,r", max_steps=100))

	@info "Defining simulation."
end

# ╔═╡ 02f590b6-b034-4ac4-afd4-76bde37e68b6
md"""
Simulation time step: $(@bind t Slider(1:length(steps), default=1))
"""

# ╔═╡ caad8138-251b-4644-9217-3e3bba49e357
render(mdp, policy; outline_state=steps[t].s, outline=false, force_blank = true, show_rewards = true)

# ╔═╡ 20f9c9e0-ac82-4a5c-b1a0-656a612789dd
"""
	P_TPT(si)
"""
function P_TPT(si::Vector{<:State})
	
	Ta = zeros(length(𝒮), length(𝒮))
	
	for i = 1:length(𝒮)
		d = transition(mdp, 𝒮[i], action(policy, 𝒮[i]))
	
		if d isa Deterministic
			Ta[i, state2idx[d.val]] = 1.0
		else
			for j = 1:length(d)
				if d.probs[j] > 0
					Ta[i, state2idx[d.vals[j]]] = d.probs[j]
				end
			end
		end
	end

	Ta2 = deepcopy(Ta[1:end-1, 1:end-1])
	
	for i = 1:size(Ta2, 1)
		for s in si
			Ta2[i, state2idx[s]] += Ta[i, end]/length(si)
		end
	end

	return Ta2
end

# ╔═╡ f03d2bfe-2479-42c8-970f-c0f894518e7e
begin
	@info "COMPUTING TPT"
	
	A_state = State[]
	B_state = State[]
	avoids = State[]

	local rewards = tuplestring2vec.(split(reward_params2, "\n"))
	
	for r in rewards
		if r[4] == 1
			push!(B_state, State(r[1], r[2]))
		elseif r[4] == 0
			push!(avoids, State(r[1], r[2]))
		end
	end

	local tpt_a = tuplestring2vec(tpt_A)
	A_state = [State(tpt_a[1], tpt_a[2])]

	P = P_TPT(A_state)
	local tpt = TPTHomog(P, 
		map(x -> state2idx[x], [A_state ; avoids]), 
		map(x -> state2idx[x], [B_state ; avoids]))
	
	tpt_stat = tpt_stationary_statistics(tpt)
	nothing
end

# ╔═╡ 9190a862-7fba-4a1d-94aa-5098ba2b5205
tpt_stat.transition_time_vanE

# ╔═╡ 9142f2e5-5c75-40b5-9ea2-d0086bd66659
let
	render(mdp, policy, 30; 
		outline=true, 
		# outline_state=State(1, 1), 
		outline_A = A_state,
		outline_B = B_state,
		outline_avoids = avoids,
		show_policy = true, 
		force_blank = true,
		show_rewards = true,
		rewards_color = :black,
		plot_vector = log.(tpt_stat.normalized_reactive_density))
end

# ╔═╡ 2dd84741-b792-414e-9692-d2faacaaf24b
let
	fij = tpt_stat.forward_current
	fi = [
		sum(fij[state2idx[si], state2idx[sj]] .* [sj.x - si.x, sj.y - si.y] for sj in 𝒮 if sj !== params.null_state) 
		for si in 𝒮 if si !== params.null_state]
	
	render(mdp, policy, 30; 
		outline=true, 
		# outline_state=State(1, 1), 
		outline_A = A_state,
		outline_B = B_state,
		outline_avoids = avoids,
		show_policy = false, 
		force_blank = true,
		show_rewards = true,
		rewards_color = :black,
		# plot_vector = log.(tpt_stat.normalized_reactive_density),
		plot_arrows = fi)
end

# ╔═╡ 32668e09-b12b-43c8-862f-b6f1a77557ec
one_based_policy!(policy) # change default initial policy from all zeros to all ones

# ╔═╡ 4d698cad-a570-4608-b6dd-20de5d7dbe33
begin
@info "VI Simulation"
vi_mdp = QuickMDP(GridWorld,
    states       = 𝒮,
    actions      = 𝒜,
    transition   = T,
    reward       = R,
    discount     = γ_vi, # custom discount for visualization of Value Iteration policy
    initialstate = 𝒮,
    isterminal   = termination);

vi_solver = ValueIterationSolver(max_iterations=vi_iterations)
vi_policy = solve(vi_solver, vi_mdp)
one_based_policy!(vi_policy)
nothing
end

# ╔═╡ f619bd9e-944d-4a0d-a2ed-022afbcfde20
render(vi_mdp, vi_policy, vi_iterations, γ_vi, show_rewards = true, outline = true)

# ╔═╡ 721ce0f2-62d7-43fd-b4c1-433adb9fa6d5
begin
	@info "Defining notebook info box."

	local blurb = md"""
	This is a [Pluto notebook](https://plutojl.org/) powered by the [Julia programming language](https://julialang.org/).

	- Written by [Gage Bonner](https://github.com/70Gage70), 2024
	- The implementation of the Grid World problem in this notebook is originally from Julia Academy's "Decision Making Under Uncertainty with POMDPs.jl". [[github]](https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty).
	"""

	local blah = ad(
		details("ABOUT", [blurb], open = false), 
	"warning")
	
	@htl """<div style="
	position: fixed; 
	right: 1rem; 
	top: 1rem; 
	padding: 1px;
	text-align: left;
	z-index: 99;
	max-width: 18%;
	background-color: var(--main-bg-color);">
	$(blah)
	</div>"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DiscreteValueIteration = "4b033969-44f6-5439-a48b-c11fa3648068"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
POMDPModelTools = "08074719-1b2a-587c-a292-00f91cc44415"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPSimulators = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TransitionPathTheory = "8e25bcd3-68d4-49ab-a21a-eececf317700"

[compat]
ColorSchemes = "~3.24.0"
Colors = "~0.12.10"
DiscreteValueIteration = "~0.4.7"
HypertextLiteral = "~0.9.5"
Latexify = "~0.16.2"
POMDPModelTools = "~0.3.13"
POMDPPolicies = "~0.4.3"
POMDPSimulators = "~0.3.14"
POMDPs = "~0.9.6"
Parameters = "~0.12.3"
Plots = "~1.40.3"
PlutoUI = "~0.7.58"
QuickPOMDPs = "~0.2.14"
TransitionPathTheory = "~0.4.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "ecf93c59996493894845b24241ab309a5fe44057"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["Tricks"]
git-tree-sha1 = "6c7d1ebb157fdf0f696698ef01946fe93c9efff4"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiscreteValueIteration]]
deps = ["LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Printf", "SparseArrays"]
git-tree-sha1 = "9ef7096a05ff90c0c7644cde929c2f6f028109f2"
uuid = "4b033969-44f6-5439-a48b-c11fa3648068"
version = "0.4.7"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

    [deps.Distances.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "bfe82a708416cf00b73a3198db0859c82f741558"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3437ade7073682993e092ca570ad68a2aba26983"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a96d5c713e6aa28c242b0d25c1347e258d6541ab"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.3+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d4f85701f569584f2cff7ba67a137d03f0cfb7d0"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "3863330da5466410782f2bffc64f3d505a6a8334"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "8e59b47b9dc525b70550ca082ce85bcd7f5477cd"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca0f6bf568b4bfc807e7537f081c81e35ceca114"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.10.0+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "656036b9ed6f942d35e536e249600bc31d0f9df8"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.0+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8f6af051b9e8ec597fa09d8885ed79fd582f33c9"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.10"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "77c3bd69fdb024d75af38713e883d0f249ce19c2"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.2+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "27d162f37cc29de047b527dab11a826dd3a650ad"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "af81a32750ebc831ee28bdaaba6e1067decef51e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[deps.POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPTools", "POMDPs", "Random", "Reexport", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "36d32d62e036ae8ebb9b8efe9e8658f902815700"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.13"

[[deps.POMDPPolicies]]
deps = ["Distributions", "LinearAlgebra", "POMDPTools", "POMDPs", "Parameters", "Random", "Reexport", "SparseArrays", "StatsBase"]
git-tree-sha1 = "bd72fbfea89a64946963518aa53097e2a1233c59"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.3"

[[deps.POMDPSimulators]]
deps = ["DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPTools", "POMDPs", "ProgressMeter", "Random", "Reexport"]
git-tree-sha1 = "3735b7a48bd892f153ab7327cb71e447e8f18e14"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.14"

[[deps.POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "6b7e405f2c1905aff6f07ee4d241fd608d289d66"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.6"

[[deps.POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9a6fe01a75a23cfb8a4d7af43f95ff3db16694f1"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.6"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "3bdfa4fa528ef21287ef659a89d686e8a1bcb1a9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.3"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.QuickPOMDPs]]
deps = ["NamedTupleTools", "POMDPTools", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "5a198408a330c4bff6e085831e542f75f4e5e38f"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.14"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TransitionPathTheory]]
deps = ["Clustering", "Graphs", "HDF5", "LinearAlgebra", "Random", "StatsBase"]
git-tree-sha1 = "fb48c7025a4017e7cd5448398b9cd5043a91d41d"
uuid = "8e25bcd3-68d4-49ab-a21a-eececf317700"
version = "0.4.4"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeTypeAbstraction", "LazyModules", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "ae67ab0505b9453655f7d5ea65183a1cd1b3cfa0"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "2.12.4"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "532e22cf7be8462035d092ff21fada7527e2c488"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.6+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─53fc4147-2716-4b36-9752-ffc3693d1c72
# ╟─acfe17fe-0467-4f0f-8fcb-bc5e6b6f01be
# ╟─b6deb2b4-39eb-48e9-8bbe-340121c3de2e
# ╟─e2377ddd-03fe-424d-82e9-8d90d2c5ac56
# ╟─a4230ab4-a655-44f2-8292-e28af4e702cb
# ╟─56cd93e8-5794-4dfa-aa66-28a550ad7849
# ╟─cd41b63d-3fc3-4fdd-871a-d0d0a3ee279d
# ╟─412f6c4c-2209-4542-9de9-d529ca5bd087
# ╟─f619bd9e-944d-4a0d-a2ed-022afbcfde20
# ╟─a8817d6e-2302-4f39-8b93-66d550ca09ef
# ╟─3590b4b0-5a05-4052-aa58-f48d3912ce77
# ╟─6864700f-4705-4c8c-97c8-24d119dbcbbc
# ╟─caad8138-251b-4644-9217-3e3bba49e357
# ╟─bc474e19-7d3e-481c-b36c-e231cd72db94
# ╟─02f590b6-b034-4ac4-afd4-76bde37e68b6
# ╟─d5df4c2e-ab77-4331-8f6d-4fd76c44bbb9
# ╟─34326f2e-b804-481a-b151-0f3529ac8b3a
# ╟─7aa7700b-490e-4929-9c78-cca7840ff86b
# ╟─a90137f2-b058-4e50-a3cf-9cae179c774d
# ╟─20f9c9e0-ac82-4a5c-b1a0-656a612789dd
# ╟─f03d2bfe-2479-42c8-970f-c0f894518e7e
# ╟─9142f2e5-5c75-40b5-9ea2-d0086bd66659
# ╟─2dd84741-b792-414e-9692-d2faacaaf24b
# ╠═9190a862-7fba-4a1d-94aa-5098ba2b5205
# ╠═2adbfa59-3d25-4819-b1f6-75b7979462c1
# ╟─5e9b28a4-50b9-4c44-8d3c-48eb72bda3a5
# ╟─b2478af7-9c04-4f01-b261-b2db30c52a5c
# ╟─becac04c-9819-42b7-811e-034440796511
# ╟─25d170e9-1c26-4058-96fc-04ab47964b51
# ╟─52b96024-9f52-4f07-926b-2297ed7dd166
# ╟─cd693f7d-1576-4c2b-a60b-67b45a03ecf2
# ╟─3cf71e78-d2b9-4267-9593-58dd46100033
# ╟─59f9c300-25ba-4193-991b-7da9df00d78d
# ╟─108efbd5-5ff8-4d17-aab4-15702dd4ef2d
# ╟─b307378e-d5eb-46a6-99b5-7032d9926c8e
# ╟─bf240f83-4fc9-4fe2-8936-b0278500204d
# ╟─e2a84ebf-a259-43c1-b512-f6c6b6e02d14
# ╠═5e5dbd4b-85f5-466a-9763-7dc936887477
# ╠═c092511d-c2e7-4b8c-8104-b4b10893cb02
# ╠═13dbf845-14a7-4c98-a1db-b3a83c9ce37c
# ╠═4a14aee4-12f1-4d55-9532-9b88e4c465f8
# ╠═c1d07fca-1fbd-4450-96b1-c829d7ad8306
# ╠═bcc5e8a3-1e3a-40cf-a306-13599a4952ac
# ╠═bc541507-61db-4084-9712-1c57d139e17f
# ╠═1303be2a-d18c-44b0-afb9-06a6b4ce5c08
# ╟─27e554ff-9861-4a41-ad65-9d5ae7727e45
# ╟─1f55df07-0f67-4ed4-b5f5-5a46fb28ab85
# ╠═49901c66-db64-48a2-b122-84d5f6b769db
# ╠═28efe3a2-36bd-48d5-9762-7410c7498856
# ╟─f7814a66-23c8-4782-ba06-755397af87db
# ╟─87a6c45e-6f3e-428e-8301-3b0c4166a84b
# ╠═6970821b-2b87-4e66-b737-512e83627998
# ╠═49b140ad-641f-436c-9492-cf3efbadd8d2
# ╠═5af462ea-781b-4509-98de-c68c29ec81fd
# ╟─e67b6b1e-0cb3-4840-b566-8ae9d3d3cdad
# ╠═48c966cb-6b79-42a6-8ff0-2fe3261f3981
# ╠═ba95fdd5-d643-4de2-8235-90e2b5651410
# ╟─32668e09-b12b-43c8-862f-b6f1a77557ec
# ╠═c85a316a-acd3-4aa3-8b65-28a332950240
# ╟─4d698cad-a570-4608-b6dd-20de5d7dbe33
# ╟─a4e1b417-60c9-475e-aa16-b2112f7b47cf
# ╠═87710f14-a7ec-4983-b105-8090c4dd1463
# ╠═92cdb96c-e636-4308-bf49-6d34246d3256
# ╠═9c139d5b-b8c9-4e00-ba5b-ee66fad73658
# ╟─aeec3109-1ebc-4edf-9ac5-eda3a30467cc
# ╟─99f2318b-b1d2-4afa-960b-eed2971abede
# ╟─b51e4f52-a3ad-479c-b4de-59261389e409
# ╟─721ce0f2-62d7-43fd-b4c1-433adb9fa6d5
# ╠═576bc4b1-3528-47b9-9345-3a5058327c28
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
