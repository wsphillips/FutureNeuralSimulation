module NeuralSimulation

using DiffEqBase
export CableSection, HHCableFunction, BEHHCable, HHCableProblem

# Placeholder geometry type, this will be it's own major portion of the package/API eventually
struct CableSection{T,N}
    radius::T
    len::T
    rl::T
    cm::T
    num::N
end

# Active membrane cable solver:
# Below is a prototype implementation of a compartmental model with active membrane dynamics,
# which resembles an unbranched, unmyelinated axon that's discretized in one dimension (length of the cable)

abstract type AbstractNeuronalAlgorithm <: DiffEqBase.DEAlgorithm end

# Eventually we might just use `ImplicitEuler` as the algorithm type, and just bake a different integrator based
# on the unique method signature
struct BackwardsEuler <: AbstractNeuronalAlgorithm end
export BackwardsEuler

abstract type AbstractNeuronalFunction{iip} <: DiffEqBase.AbstractDiffEqFunction{iip} end

# A NeuronalSystem containing equations should be rewritten into more efficient forms for certain methods.
# The primary case being rewrites for solving backwards Euler in fewer steps (we don't have to do fixed-point iteration)
struct HHCableFunction{iip, F1, F2} <: AbstractNeuronalFunction{iip}
    f1::F1  # in-place update of coefficients for Hines matrix
    f2::F2  # in-place Channel gating dynamics
    function HHCableFunction{iip}(f1, f2) where {iip}
        new{iip, typeof(f1), typeof(f2)}(f1,f2)
    end
end

abstract type AbstractNeuronalIntegrator{Alg, IIP, U, T} <: DiffEqBase.DEIntegrator{Alg, IIP, U, T} end

mutable struct BackwardsEulerHHCableIntegrator{IIP, S, T, P, F, HinesMatrix} <: AbstractNeuronalIntegrator{BackwardsEuler, IIP, S, T}
    f::F     # equations
    uprev::S # Previous state
    u::S     # Current state 
    tprev::T # Previous time step
    t::T     # Current time step
    t0::T    # Initial time step, only for re-initialization
    dt::T    # Step size
    p::P     # Parameters container
    H::HinesMatrix # Hines matrix
    function BackwardsEulerHHCableIntegrator(f::HHCableFunction, uprev, u, tprev, t, t0, dt, p, H)
        new{isinplace(f), typeof(u), typeof(t), typeof(p), typeof(f), typeof(H)}(
            f, uprev, u, tprev, t, t0, dt, p, H
        )
    end
end

const BEHHCable = BackwardsEulerHHCableIntegrator
DiffEqBase.isinplace(::HHCableFunction{IIP}) where {IIP} = IIP
DiffEqBase.isinplace(::BEHHCable{IIP}) where {IIP} = IIP

abstract type AbstractNeuronalProblem{uType, tType, isinplace} <: DiffEqBase.DEProblem end

struct HHCableProblem{uType, tType, isinplace, P, F} <: AbstractNeuronalProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P # constants to be supplied as second arg of `f`
    function HHCableProblem(f::AbstractNeuronalFunction, u0, tspan, p)
        new{typeof(u0), typeof(tspan), DiffEqBase.isinplace(f), typeof(p), typeof(f)}(f, u0, tspan, p)
    end
end

# Initialization

function DiffEqBase.solve(prob::AbstractNeuronalProblem, args...; kwargs...)
    __solve(prob,args...; kwargs...)
end

function DiffEqBase.init(prob::AbstractNeuronalProblem, args...; kwargs...)
    __init(prob,args...; kwargs...)
end

function DiffEqBase.__init(prob::HHCableProblem, alg::BackwardsEuler; dt = error("dt is required for this algorithm"))
    hhcable_init(prob.f,
                 DiffEqBase.isinplace(prob),
                 prob.u0,
                 prob.tspan[1],
                 dt,
                 prob.p)
end

function DiffEqBase.__solve(prob::HHCableProblem, alg::BackwardsEuler; dt = error("dt is required for this algorithm"))

    u0    = prob.u0
    tspan = prob.tspan
    ts    = Array(tspan[1]:dt:tspan[2])
    n     = length(ts)
    # It looks like memory for the result isn't pre-allocated. Only an array of pointers (`undef`)
    us    = Vector{typeof(u0)}(undef, n)

    # since we're using fixed time-step, allocate an array for history of time series results
    us[1] = copy(u0)
    for j in 2:n
        us[j] = similar(u0)
    end
    # construct integrator (holds pre-allocated state/parameter data needed for each step)
    integ = hhcable_init(prob.f, #=DiffEqBase.isinplace(prob)=# true, prob.u0, prob.tspan[1], dt, prob.p)

    # step through time series and copy new state vector to history (`us`) each step
    for i in 2:n
        step!(integ)
        us[i] .= integ.u
    end

    # for now just return the vector series of Vm values
    # sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)
    # return sol
    return us
end

@inline function hhcable_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P) where
    {F, P, T, S<:AbstractArray{T}}

    integ = HHCable{IIP, S, T, P, F}(f, copy(u0), copy(u0), t0, t0, t0, dt, p)
    return integ
end

# Solves an unbranched cable (tridiagonal matrix); needs modification for generalization to branched cable case
# In branched structures, references to the parent compartment (idx - 1) are substituted with a lookup table
# NOTE: This solver is independent of the DE method (its just slightly-modified LAPACK--see `dgtsv()`)
function thomas!(Hines, V)
    #Backward pass
    for i in N:-1:2
        factor = Hines.U[i] / Hines.D[i]
        Hines.D[i-1] -= factor * Hines.L[i] 
        V[i-1] -= factor * V[i] 
    end

    # Root solve
    V[1] /= Hines.D[1]

    # Forward pass
    for j in 2:N
        V[j] -= Hines.L[j]*V[j-1]
        V[j] /= Hines.D[j]
    end

    return nothing
end

# Stepping
# In-place stepping
@inline function DiffEqBase.step!(integ::BEHHCable{true, S, T}) where {T, S}
    
    integ.uprev        .= integ.u
    f                   = integ.f
    p                   = integ.p
    t                   = integ.t
    dt                  = integ.dt
    u                   = integ.u
    update_hines!       = f.f_hines

    integ.tprev = t
    integ.t += dt

    hines!(Hines, u, p, dt)
    # Solve Hines for membrane voltage
    thomas!(Hines, view(u, :, 1))
    # Update dimensionless channel gating vars
    stepgates!(u, p, t)

    return nothing
end

# TODO: Interpolation
function (integ::BEHHCable)(t::T) where T
end

end # module
