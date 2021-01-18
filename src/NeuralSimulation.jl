module NeuralSimulation

using DiffEqBase

# Active membrane cable solver:
# Below is a prototype implementation of a compartmental model with active membrane dynamics,
# which resembles an unbranched, unmyelinated axon that's discretized in one dimension (length of the cable)

# Right now, I'm using a hand-rolled version of backwards Euler. We should have freedom to swap

# Hodgkin-Huxley channel gating functions
αn(V) = V == -55.0 ? 0.1 : (0.01*(V + 55))/(1 - exp(-(V + 55)/10))
βn(V) = 0.125 * exp(-(V + 65)/80)
αm(V) = V == -40.0 ? 1.0 : (0.1*(V + 40))/(1 - exp(-(V + 40)/10))
βm(V) = 4*exp(-(V + 65)/18)
αh(V) = 0.07*exp(-(V+65)/20)
βh(V) = 1/(1 + exp(-(V + 35)/10))

# Steady-state channel gating
m∞(V) = αm(V)/(αm(V) + βm(V))
h∞(V) = αh(V)/(αh(V) + βh(V))
n∞(V) = αn(V)/(αn(V) + βn(V))

function stepm!(m, V, dt)
    @. m = (m + αm(V)*dt)/(1+ (αm(V) + βm(V))*dt)
end

function steph!(h, V, dt)
    @. h = (h + αh(V)*dt)/(1+ (αh(V) + βh(V))*dt)
end

function stepn!(n, V, dt)
    @. n = (n + αn(V)*dt)/(1+ (αn(V) + βn(V))*dt)
end

function update_gates!(V, m, h, n, dt)
    stepm!(m, V, dt)
    steph!(h, V, dt)
    stepn!(n, V, dt)
end

abstract type AbstractNeuronalAlgorithm <: DiffEqBase.DEAlgorithm end
struct SimpleActiveCable <: AbstractNeuronalAlgorithm end
export SimpleActiveCable

abstract type AbstractNeuronalFunction{iip} <: DiffEqBase.AbstractDiffEqFunction{iip} end

struct SimpleActiveCableFunction{iip, F1, F2, S} <: AbstractNeuronalFunction{iip}
    
    f1::F1  # Cable equation (function) with HH
    f2::F2  # Channel gating dynamics
    
    # intermediates for cable equation solution
    L::S     # subdiagonal
    D::S     # Diagonal
    U::S     # super-diagonal
    # ignoring other typical fields used in DiffEq function types for now
end

abstract type AbstractNeuronalIntegrator{Alg, IIP, U, T} <: DiffEqBase.DEIntegrator{Alg, IIP, U, T} end

mutable struct SimpleActiveCableIntegrator{IIP, S, T, P, F} <: AbstractNeuronalIntegrator{SimpleActiveCable, IIP, S, T}
    f::F     # Cable equations
    uprev::S # Previous state
    u::S     # Current state 
    tprev::T # Previous time step
    t::T     # Current time step
    t0::T    # Initial time step, only for re-initialization
    dt::T    # Step size
    p::P     # Parameters container
end

const SACable = SimpleActiveCableIntegrator
DiffEqBase.isinplace(::SACable{IIP}) where {IIP} = IIP

abstract type AbstractNeuronalProblem{uType, tType, isinplace} <: DiffEqBase.DEProblem end

struct SimpleActiveCableProblem{uType, tType, isinplace, P, F, K} <: AbstractNeuronalProblem{uType, tType, isinplace}
    f::F
    u0::uType
    tspan::tType
    p::P # constants to be supplied as second arg of `f`
    kwargs::K # DiffEq says this is `a callback to be applied to every solver which uses the problem`
              # I'm not sure what's meant by that atm
    function SimpleActiveCableProblem{iip}(f::AbstractNeuronalFunction{iip}, u0, tspan, p; kwargs...) where {iip}
        new{typeof(u0), typeof(tspan), isinplace(f), typeof(p), typeof(f), typeof(kwargs)}(f, u0, tspan, p, kwargs)
    end
end

# Initialization

function DiffEqBase.solve(prob::AbstractNeuronalProblem, args...; kwargs...)
    __solve(prob,args...; kwargs...)
end

function DiffEqBase.__init(prob::SimpleActiveCableProblem, alg::SimpleActiveCable; dt = error("dt is required for this algorithm"))
    simpleactivecable_init(prob.f,
                   DiffEqBase.isinplace(prob),
                   prob.u0,
                   prob.tspan[1],
                   dt,
                   prob.p)
end

function DiffEqBase.__solve(prob::SimpleActiveCableProblem, alg::SimpleActiveCable; dt = error("dt is required for this algorithm"))

    u0    = prob.u0
    tspan = prob.tspan
    ts    = Array(tspan[1]:dt:tspan[2])
    n     = length(ts)
    # It looks like memory for the result isn't pre-allocated. Only an array of pointers (`undef`)
    us    = Vector{typeof(u0)}(undef, n)

    # allocate array for history of time series results
    # note that array copying is different for static arrays (see original SimpleDiffEq.jl RK4 implementation)
    @inbounds us[1] = copy(u0)

    # construct integrator (holds pre-allocated state/parameter data needed for each step)
    integ = simpleactivecable_init(prob.f, DiffEqBase.isinplace(prob), prob.u0, prob.tspan[1], dt, prob.p)

    # step through time series and copy new state vector to history (`us`) each step
    for i = 1:n-1
        step!(integ)
        us[i+1] = copy(integ.u) # I think this is allocating? Why not copyto?
    end

    # We might need a dispatch for build_solution too?
    #=
    sol = DiffEqBase.build_solution(prob, alg, ts, us, calculate_error = false)

    return sol
    =#
    # for now just return the vector series of Vm values
    return us
end

@inline function simpleactivecable_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P) where
    {F, P, T, S<:AbstractArray{T}}

    integ = SACable{IIP, S, T, P, F}(f, _copy(u0), _copy(u0), t0, t0, t0, dt, p)
    return integ
end

# Stepping
# In-place stepping

@inline function DiffEqBase.step!(integ::SACable{true, S, T}) where {T, S}
    integ.uprev       .= integ.u
    tmp                = integ.tmp
    f!                 = integ.f
    p                  = integ.p
    t                  = integ.t
    dt                 = integ.dt
    uprev              = integ.uprev
    u                  = integ.u

    f!(du, u, p, t + dt)

    integ.tprev = t
    integ.t += dt

    # First solve Cable equation
    @. f.D = 1 + (a*dt)/(rl*cm*dx^2) + (dt/cm)*(gna*m^3*h + gk*n^4 + gl)

    # in first compartment we inject a 1 nA current
    u[1] = u[1] + (dt/cm)*(gna*m[1]^3*h[1]*Ena + gk*n[1]^4*Ek + gl*El) + (Iapp*dt)/(pi*a*cm*dx)
    u[2:end] .= @. u[2:end] + (dt/cm)*(gna*m[2:end]^3*h[2:end]*Ena + gk*n[2:end]^4*Ek + gl*El)

    #Backward pass
    for i in N:-1:2
        factor = f.U[i] / D[i]
        f.D[i-1] -= factor * f.L[i] 
        u[i-1] -= factor * u[i] 
    end
    # Root solve
    u[1] /= f.D[1]
    # Forward pass
    for j in 2:N
        u[j] -= f.L[j]*u[j-1]
        u[j] /= f.D[j]
    end
    
    # Update dimensionless channel gating vars
    update_gates!(u, m, h, n, dt)

    return nothing
end

# Allocating step
@inline function DiffEqBase.step!(integ::SACable{false, S, T}) where {T, S}
    integ.uprev = integ.u
    f           = integ.f
    p           = integ.p
    t           = integ.t
    dt          = integ.dt
    uprev       = integ.uprev

    result = f(integ.u, p, t + dt)

    integ.tprev = t
    integ.t += dt

    return nothing
end

# Interpolation
# I think we can skip this for now...
#=
@inline @muladd function (integ::SRK4)(t::T) where T
    t₁, t₀, dt = integ.t, integ.tprev, integ.dt

    y₀ = integ.uprev
    y₁ = integ.u
    ks = integ.ks
    Θ  = (t - t₀)/dt

    # Hermite interpolation.
    @inbounds if !isinplace(integ)
        u = (1-Θ)*y₀ + Θ*y₁ + Θ*(Θ-1)*( (1-2Θ)*(y₁-y₀) +
                                        (Θ-1)*dt*ks[1] +
                                        Θ*dt*ks[5])
        return u
    else
        u = similar(y₁)
        for i in 1:length(u)
            u[i] = (1-Θ)*y₀[i] + Θ*y₁[i] + Θ*(Θ-1)*( (1-2Θ)*(y₁[i]-y₀[i])+
                                                     (Θ-1)*dt*ks[1][i] +
                                                     Θ*dt*ks[5][i])
        end

        return u
    end
end
=#
end # module
