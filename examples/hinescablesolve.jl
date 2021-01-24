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

function step!(L, D, U, RHS, a, dt, dx, rl, gl, gna, gk, El, Ena, Ek, m, h, n, N, Iapp)
    # First solve Cable equation
    @. D = 1 + (a*dt)/(rl*cm*dx^2) + (dt/cm)*(gna*m^3*h + gk*n^4 + gl)

    # in first compartment we inject a 1 nA current
    RHS[1] = RHS[1] + (dt/cm)*(gna*m[1]^3*h[1]*Ena + gk*n[1]^4*Ek + gl*El) + (Iapp*dt)/(pi*a*cm*dx)
    RHS[2:end] .= @. RHS[2:end] + (dt/cm)*(gna*m[2:end]^3*h[2:end]*Ena + gk*n[2:end]^4*Ek + gl*El)

    #Backward pass
    for i in N:-1:2
        f = U[i] / D[i]
        D[i-1] -= f * L[i] 
        RHS[i-1] -= f * RHS[i] 
    end

    # Root solve
    RHS[1] /= D[1]

    # Forward pass
    for j in 2:N
        RHS[j] -= L[j]*RHS[j-1]
        RHS[j] /= D[j]
    end
    
    # Update dimensionless channel gating vars
    update_gates!(RHS, m, h, n, dt)
end

dt = 0.025 # ms
Iapp = 1.0e-6 # mA (1 nA)
cable_length = 2000.0e-3 # 2000 µm length
N = 2000
dx = cable_length/N
a   = 2.0e-3 # µm in mm
area = 2*pi*a*dx

rl = 1.0e6 # 1 kΩ in milliohms*mm
gl  = 0.003 #mS/mm^2
gna = 1.2 # mS/mm^2 (modified from 1.2 mS)
gk  = .36 # mS/mm^2
El  = -54.4 # mV
Ena = 50.0 # mV
Ek  = -77.0 # mV
cm  = 1.0e-5 # mF/mm^2
Vinit = -55.0 # mV

Li = -(a*dt)/(2*rl*cm*dx^2)
LN = -(a*dt)/(rl*cm*dx^2)
Ui = Li
U1 = LN

L = Vector{Float64}(undef, N)
U = Vector{Float64}(undef, N)
D = Vector{Float64}(undef, N)
RHS = fill(Vinit,(N))
m = fill(m∞(Vinit), (N))
h = fill(h∞(Vinit), (N))
n = fill(n∞(Vinit), (N))
L[1] = 0; L[N] = LN; L[2:end-1] .= Li
U[1] = U1; U[N] = 0; U[2:end-1] .= Ui
history = Vector{Vector{Float64}}()

@time begin
    for i in 1:8000
        step!(L, D, U, RHS, a, dt, dx, rl, gl, gna, gk, El, Ena, Ek, m, h, n, N, Iapp)
    end 
end

using Plots

plot(RHS)
