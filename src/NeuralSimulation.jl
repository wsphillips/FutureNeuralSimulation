module NeuralSimulation

using DiffEqBase
export CableSection, HHCableFunction, BEHHCable, HHCableProblem, HinesMatrix, init, solve, step!

abstract type AbstractNeuronalAlgorithm <: DiffEqBase.DEAlgorithm end

# Eventually we might just use `ImplicitEuler` as the algorithm type, and just bake a different integrator based
# on the unique method signature
struct BackwardsEuler <: AbstractNeuronalAlgorithm end
export BackwardsEuler

# Placeholder geometry type, this will be it's own major portion of the package/API eventually
abstract type AbstractGeometry end

struct CableSection{T,N} <: AbstractGeometry
    radius::T
    len::T
    rl::T
    cm::T
    num::N
end

mutable struct HinesMatrix{S, Alg}
    L::S
    D::S
    U::S
end

# TODO: other constructors for more elaborate tree structure types
# this version is specific to backwards Euler
function HinesMatrix(cable::CableSection, alg::BackwardsEuler, dt)
        rl = cable.rl
        a = cable.radius
        cm = cable.cm
        n = cable.num
        dx = cable.len/n

        L = Vector{typeof(cable.len)}(undef, n)
        D = similar(L)
        U = similar(L)

        # L and U are static
        L[1] = 0.0
        L[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        L[end] = -(a*dt)/(rl*cm*dx^2)
        
        U[1] = -(a*dt)/(rl*cm*dx^2)
        U[2:end-1] .= -(a*dt)/(2*rl*cm*dx^2)
        U[end] = 0.0

        return HinesMatrix{typeof(L), typeof(alg)}(L, D, U)
end

abstract type AbstractNeuronalFunction{iip} <: DiffEqBase.AbstractDiffEqFunction{iip} end

# A NeuronalSystem containing equations should be rewritten into more efficient forms for certain methods.
# e.g. rewrites for solving backwards Euler in fewer steps (as we have done below)
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
end

const BEHHCable = BackwardsEulerHHCableIntegrator
DiffEqBase.isinplace(::HHCableFunction{IIP}) where {IIP} = IIP
DiffEqBase.isinplace(::BEHHCable{IIP}) where {IIP} = IIP

abstract type AbstractNeuronalProblem{uType, tType, isinplace} <: DiffEqBase.DEProblem end

struct HHCableProblem{uType, tType, isinplace, P, F, G} <: AbstractNeuronalProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P # constants to be supplied as second arg of `f`
    geometry::G
    function HHCableProblem(f::AbstractNeuronalFunction, cable::AbstractGeometry, u0, tspan, p)
        new{typeof(u0), typeof(tspan), DiffEqBase.isinplace(f), typeof(p), typeof(f), typeof(cable)}(f, u0, tspan, p, cable)
    end
end

DiffEqBase.isinplace(::HHCableProblem{U, T, IIP, P, F}) where {U, T, IIP, P, F} = IIP

# Initialization

function DiffEqBase.solve(prob::AbstractNeuronalProblem, args...; kwargs...)
    DiffEqBase.__solve(prob,args...; kwargs...)
end

function DiffEqBase.init(prob::AbstractNeuronalProblem, args...; kwargs...)
    DiffEqBase.__init(prob,args...; kwargs...)
end

function DiffEqBase.__init(prob::HHCableProblem, alg::BackwardsEuler; dt = error("dt is required for this algorithm"))
    cable = prob.geometry
    Hines = HinesMatrix(cable, alg, dt)
    hhcable_init(prob.f,
                 DiffEqBase.isinplace(prob),
                 prob.u0,
                 prob.tspan[1],
                 dt,
                 prob.p,
                 Hines)
end

function DiffEqBase.__solve(prob::HHCableProblem, alg::BackwardsEuler; dt = error("dt is required for this algorithm"))
    u0    = prob.u0
    tspan = prob.tspan
    ts    = Array(tspan[1]:dt:tspan[2])
    n     = length(ts)
    us    = Vector{typeof(u0)}(undef, n)
    cable = prob.geometry
    # With fixed-step methods we can pre-allocate the history of results
    us[1] = copy(u0)
    for j in 2:n
        us[j] = similar(u0)
    end
    integ = DiffEqBase.__init(prob, alg, dt = dt)
    for i in 2:n
        step!(integ)
        us[i] .= integ.u
    end
    # for now just return the vector series of Vm values
    # sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)
    # return sol
    return us
end

@inline function hhcable_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P, H::HinesMatrix) where
    {F, P, T, S<:AbstractArray{T}}
    integ = BEHHCable{IIP, S, T, P, F, typeof(H)}(f, copy(u0), copy(u0), t0, t0, t0, dt, p, H)
    return integ
end

# Solves an unbranched cable (tridiagonal matrix); needs modification for generalization to branched cable case
# In branched structures, references to the parent compartment (idx - 1) are substituted with a lookup table
# NOTE: This solver is independent of the DE method (its just slightly-modified LAPACK--see `dgtsv()`)
function thomas!(Hines, V)
    N = length(V)
    L = Hines.L
    D = Hines.D
    U = Hines.U
    #Backward pass
    for i in N:-1:2
        f = U[i] / D[i]
        D[i-1] -= f * L[i] 
        V[i-1] -= f * V[i] 
    end
    # Root solve
    V[1] /= D[1]
    # Forward pass
    for j in 2:N
        V[j] -= L[j]*V[j-1]
        V[j] /= D[j]
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
    update_hines!       = f.f1
    solve_gates!        = f.f2
    Hines               = integ.H
    integ.tprev = t
    integ.t += dt
    # Update coefficients of Hines matrix
    update_hines!(integ.H, integ.u, p, dt)
    # Solve Hines for membrane voltage
    thomas!(integ.H, view(integ.u, :, 1))
    # Update dimensionless channel gating vars
    solve_gates!(integ.u, p, dt)
    return nothing
end

# TODO: Interpolation
function (integ::BEHHCable)(t::T) where T
end

end # module
