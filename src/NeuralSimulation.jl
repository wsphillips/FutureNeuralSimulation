module NeuralSimulation

using DiffEqBase
export CableSection, HHCableFunction, BEHHCable, HHCableProblem

# Placeholder geometry type, this will be it's own major portion of the package/API eventually
struct CableSection
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
struct HHCableFunction{iip, F1, F2, BEF1, BEF2, S} <: AbstractNeuronalFunction{iip}
    f1::F1  # Voltage equation (du, u, p, t)
    f2::F2  # Channel gating dynamics (du, u, p, t)
    LDUsolve::BEF1 # Voltage backwards Euler tridiagonal solve (u, p, dt)
    itrgates::BEF2 # Channel gates Backwards euler iterable approximation (u, p, dt)
    # ignoring other typical fields used in DiffEq function types for now
end

function SimpleActiveCableFunction{iip, F1, F2, S}()

abstract type AbstractNeuronalIntegrator{Alg, IIP, U, T} <: DiffEqBase.DEIntegrator{Alg, IIP, U, T} end

mutable struct BackwardsEulerHHCableIntegrator{IIP, S, T, P, F} <: AbstractNeuronalIntegrator{SimpleActiveCable, IIP, S, T}
    f::F     # equations
    uprev::S # Previous state
    u::S     # Current state 
    tprev::T # Previous time step
    t::T     # Current time step
    t0::T    # Initial time step, only for re-initialization
    dt::T    # Step size
    p::P     # Parameters container
end

const BEHHCable = BackwardsEulerHHCableIntegrator
DiffEqBase.isinplace(::BEHHCable{IIP}) where {IIP} = IIP

abstract type AbstractNeuronalProblem{uType, tType, isinplace} <: DiffEqBase.DEProblem end

struct HHCableProblem{uType, tType, isinplace, P, F, K} <: AbstractNeuronalProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P # constants to be supplied as second arg of `f`
    kwargs::K # DiffEq says this is `a callback to be applied to every solver which uses the problem`
              # I'm not sure what's meant by that atm
    function HHCableProblem{iip}(f::AbstractNeuronalFunction{iip}, u0, tspan, p; kwargs...) where {iip}
        new{typeof(u0), typeof(tspan), isinplace(f), typeof(p), typeof(f), typeof(kwargs)}(f, u0, tspan, p, kwargs)
    end
end

# Initialization

function DiffEqBase.solve(prob::AbstractNeuronalProblem, args...; kwargs...)
    __solve(prob,args...; kwargs...)
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
    integ = hhcable_init(prob.f, DiffEqBase.isinplace(prob), prob.u0, prob.tspan[1], dt, prob.p)

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

# Stepping
# In-place stepping

@inline function DiffEqBase.step!(integ::BEHHCable{true, S, T}) where {T, S}
    
    integ.uprev       .= integ.u
    stepvoltage!       = integ.f1
    stepgates!         = integ.f2
    p                  = integ.p
    t                  = integ.t
    dt                 = integ.dt
    u                  = integ.u

    integ.tprev = t
    integ.t += dt

    # Update membrane voltage
    stepvoltage!(du, u, p, t)
    # Update dimensionless channel gating vars
    stepgates!(du, u, p, t)

    return nothing
end

# TODO: Interpolation
function (integ::BEHHCable)(t::T) where T
end

end # module
