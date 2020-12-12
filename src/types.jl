
abstract type Geometry end
abstract type Point <: Geometry end
abstract type Sphere <: Geometry end
abstract type Cylinder <: Geometry end


# Dynamics types describe inherent non-linearities/dynamics
abstract type Dynamics end
abstract type HodgkinHuxley <: Dynamics end
abstract type Izhikevich <: Dynamics end
abstract type MorrisLecar <: Dynamics end

const SectionID = UInt64

struct Compartment{G <: Geometry, D <: Dynamics, S <: Synapse}
    id::CompartmentID
    dims
end

abstract type AbstractNeuron end

abstract type SingleCompartment <: AbstractNeuron end

struct MultiCompartment <: AbstractNeuron
    compartments::IdDict{CompartmentID, Compartment}
    morphology::SimpleGraph # e.g. from LightGraphs.jl or some acyclic graph type
    synapses::Vector{CompartmentID}
end

abstract type ArtificialNN <: AbstractNeuron end


chnl = @channel begin
   # name, describe, and register a custom channel model 
end

dyn = @dynamics begin
    # name and describe a set of membrane properties/non-linear dynamics
end

neuron = @neuron begin
    
end


