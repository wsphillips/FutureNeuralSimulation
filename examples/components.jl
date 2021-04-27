
using ModelingToolkit, OrdinaryDiffEq, Unitful

using Unitful.DefaultSymbols
import Unitful: Voltage, mV, mS, cm, µF, mF
import Base: show, display

@derived_dimension SpecificConductance 𝐈^2*𝐋^-4*𝐌^-1*𝐓^3 # conductance per unit area
@derived_dimension SpecificCapacitance 𝐈^2*𝐋^-4*𝐌^-1*𝐓^4 # capacitance per unit area

@parameters t
const D = Differential(t)

# Ion species -- there could be more to do here (e.g. non-specific cation/anion etc)
abstract type Ion end
abstract type Calcium <: Ion end
abstract type Sodium <: Ion end
abstract type Potassium <: Ion end
abstract type Chloride <: Ion end
abstract type Leak <: Ion end # placeholder for non-specific ion conductance

# Convenience aliases
const Ca = Calcium
const Na = Sodium
const K = Potassium
const Cl = Chloride

abstract type AbstractGateModel end

struct SSTau <: AbstractGateModel
    g∞::Function
    τg::Function
    p::Real # exponent
end

struct AlphaBeta <: AbstractGateModel
    α::Function
    β::Function
    p::Real # exponent
end

AlphaBeta(α, β; exponent = 1) = AlphaBeta(α,β,exponent)
SSTau(g∞,τg; exponent = 1) = SSTau(g∞, τg, exponent)

# Steady-state channel gating
steady_state(gate::AlphaBeta, V) = gate.α(V)/(gate.α(V) + gate.β(V))
steady_state(gate::SSTau, V) = gate.g∞(V)

# Differential equations
function gate_equation(model::AlphaBeta, gate::Num, V::Num)
    α, β = model.α, model.β
    D(gate) ~ α(V) * (1 - gate) - β(V)*gate
end

function gate_equation(model::SSTau, gate::Num, V::Num)
    τ, g∞ = model.τg, model.g∞
    D(gate) ~ (1/τ(V))*(g∞(V) - gate)
end

# Conductance types
abstract type AbstractConductance end

# Eventually define more concrete types
abstract type ChemicalSynapse <: AbstractConductance end
abstract type ElectricalSynapse <: AbstractConductance end
abstract type ExchangePump <: AbstractConductance end

# This should be better (e.g. provide a callback function Iapp(t) or similar)
struct Stimulus <: AbstractConductance
    Iapp::Real
end

function Stimulus(; amps = 800pA)
    amp_val = ustrip(Float64, mA, amps)
    return Stimulus(amp_val)
end

struct PassiveChannel{I} <: AbstractConductance where {I <: Ion}
    gbar::Real
end

# FIXME: This is lazy--likely define this as a trivial ODESystem like
# "Pin" or "Ground" in the MTK tutorial
function PassiveChannel{I}(; max_g::SpecificConductance) where {I<:Ion}
    gbar_val = ustrip(Float64, mS/cm^2, max_g)
    return PassiveChannel{I}(gbar_val)
end

struct VoltageGatedChannel{I} <: AbstractConductance where {I <: Ion}
    sys::ODESystem
end

struct VoltageIonGatedChannel{I,C} <: AbstractConductance where {I <: Ion, C <: Ion}
    sys::ODESystem
end

# Return ODESystem pretty printing for our wrapper types
Base.show(io::IO, ::MIME"text/plain", x::VoltageGatedChannel) = Base.display(x.sys)

const ChordConductance{I} = Union{VoltageGatedChannel{I},PassiveChannel{I}}

ion_type(x::ChordConductance{I}) where {I<:Ion} = I

function VoltageGatedChannel{I}(gate_models::Vector{<:AbstractGateModel}; max_g::SpecificConductance,
                           Vinit::Voltage = -65mV, name) where {I<:Ion}
    
    # Strip off units
    gbar_val = ustrip(Float64, mS/cm^2, max_g)
    Vinit_val = ustrip(Float64, mV, Vinit)
    
    # Where gates are m, h etc and g(t) is channel conductance at time t
    states = @variables V(t), gates[1:length(gate_models)](t), g(t)
    params = @parameters gbar # ̄g = maximal conductance
    eqs = [gate_equation(x, gates[i], V) for (i,x) in enumerate(gate_models)]
    
    # g*x^a*y^b... 
    total_g = g ~ gbar * *((gates[i]^(x.p) for (i,x) in enumerate(gate_models))...) 
    eqs = [eqs..., total_g] 

    defaultmap = [V => Vinit_val, gbar => gbar_val,
                  (gates[i] => steady_state(x, Vinit_val) for (i,x) in enumerate(gate_models))...]

    system = ODESystem(eqs, t, [states...], [params...]; name = name, defaults = defaultmap)

    return VoltageGatedChannel{I}(system)
end

function VoltageIonGatedChannel{I,C}(gate_models::Vector{<:AbstractGateModel}; max_g::SpecificConductance,
                                     Vinit::Voltage = -65mV, name) where {I<:Ion, C<:Ion}


abstract type AbstractNeuron end

struct Neuron <: AbstractNeuron
    sys::ODESystem
end

function Neuron(channels::Vector{<:AbstractConductance}, reversals; cm::SpecificCapacitance = 1µF/cm^2) 
    
    #Strip off units
    area = 1.0 # for now assume surface area 1.0
    cm_val = ustrip(Float64, mF/cm^2, cm)
    rev_vals = IdDict(reversals) 
    states = @variables V(t), I[1:length(channels)](t)
    params = @parameters Cm E[1:length(reversals)]
    
    ions = unique([ion_type(x) for x in channels])
    eq = [D(V) ~ -sum(chan.sys.g * area * (V - E[]))]

end

