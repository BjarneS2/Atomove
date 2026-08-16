module TweezerControlsMSA

using JuMP
import Ipopt
using HDF5
using ..Types3D
using ..Models3D
using ..ThermalSampling3D
using ..InitialGuess3D
using ..Basis3D
using Random

export MSAConfig3D, MSAControlResult3D
export optimize_controls3d_msa, load_msa_guess_from_file

Base.@kwdef struct MSAConfig3D
    N_modes_s::Int = 10
    N_modes_ua::Int = 8
    ua_mode::Symbol = :fourier
    lambda_slosh::Float64 = 0.0
    lambda_mse::Float64 = 0.0
    use_primary::Bool = true
    use_secondary::Bool = true
end

struct MSAControlResult3D
    protocol::ControlProtocol3D
    samples::Vector{NamedTuple}
    x_traj::Matrix{Float64}
    y_traj::Matrix{Float64}
    z_traj::Matrix{Float64}
    vx_traj::Matrix{Float64}
    vy_traj::Matrix{Float64}
    vz_traj::Matrix{Float64}
    coeff_s::Vector{Float64}
    coeff_ua::Vector{Float64}
    delta_s0::Float64
    seed::Union{Nothing,Int}
    n_samples::Int
    termination_status::String
    objective_value::Float64
end

function load_msa_guess_from_file(path::AbstractString, p::TweezerParams3D)
    return load_guess_from_file(path, p)
end

function optimize_controls3d_msa(
    p::TweezerParams3D,
    msa::MSAConfig3D=MSAConfig3D();
    guess::Union{Nothing,InitialG3D}=nothing,
    bounds::Union{Nothing,ControlBounds3D}=nothing,
    silent::Bool=true,
    max_iter::Int=4000,
    print_level::Int=5,
    hessian_approximation::Bool=true,
    n_samples::Int=10,
    seed::Union{Nothing,Int}=nothing,
    consts::PhysicalConstants3D=default_constants3d(),
)
    bounds === nothing && (bounds = default_bounds3d(p))
    seed !== nothing && Random.seed!(seed)
    scales = compute_scales3d_full(p; consts=consts)
    g = scales.g_dimless

    ex, ey, L = transport_direction(p)

    println("Sampling $n_samples thermal initial conditions (3D, only trapped)...")
    samples = [sample_initial_conditions3d(p; consts=consts, check_trapped=true)
               for _ in 1:n_samples]

    println("  x range: [$(minimum(s.x for s in samples)), $(maximum(s.x for s in samples))]")
    println("  z range: [$(minimum(s.z for s in samples)), $(maximum(s.z for s in samples))]")

    n = p.n
    w = p.w
    w_a = p.w * p.w_aux_factor
    zR = p.zR
    zR_a = p.zR_aux
    w2 = w^2
    wa2 = w_a^2
    cz = 0.0

    Ns = 2 * msa.N_modes_s
    Nua = msa.ua_mode === :square ? 0 : 2 * msa.N_modes_ua
    Φ_s = build_s_basis(n, msa.N_modes_s)
    Φ_ua = msa.ua_mode === :square ? nothing : build_ua_basis(n, msa.N_modes_ua)
    τ = collect(range(0.0, 1.0, length=n))

    model = Model(Ipopt.Optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "print_level", print_level)
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-5)
    hessian_approximation && set_optimizer_attribute(model, "hessian_approximation", "limited-memory")

    @variable(model, bounds.T_min_fraction * p.maxT <= T <= p.maxT)
    @expression(model, dt, T / (n - 1))

    @variable(model, coeff_s[1:Ns])
    @variable(model, -bounds.u_margin_w * w <= delta_s0 <= bounds.u_margin_w * w)

    @expression(model, s[j=1:n],
        L * τ[j] + delta_s0 * (1.0 - τ[j]) + sum(Φ_s[j, k] * coeff_s[k] for k in 1:Ns)
    )

    if msa.ua_mode === :square
        @variable(model, ua_amp)
        set_lower_bound(ua_amp, bounds.ua_min)
        set_upper_bound(ua_amp, bounds.ua_max)
        @expression(model, ua[j=1:n], ua_amp)
    else
        @variable(model, coeff_ua[1:Nua])
        @expression(model, ua[j=1:n], sum(Φ_ua[j, k] * coeff_ua[k] for k in 1:Nua))
    end

    @expression(model, ux[j=1:n], p.x_start + s[j] * ex)
    @expression(model, uy[j=1:n], p.y_start + s[j] * ey)

    u_margin = bounds.u_margin_w * w
    for j in 1:n
        @constraint(model, s[j] >= -u_margin)
        @constraint(model, s[j] <= L + u_margin)
        if msa.ua_mode !== :square
            @constraint(model, ua[j] >= bounds.ua_min)
            @constraint(model, ua[j] <= bounds.ua_max)
        end
    end

    x_margin = bounds.r_margin_w * w
    @variable(model, x_s[1:n_samples, 1:n])
    @variable(model, y_s[1:n_samples, 1:n])
    @variable(model, z_s[1:n_samples, 1:n])
    @variable(model, vx_s[1:n_samples, 1:n])
    @variable(model, vy_s[1:n_samples, 1:n])
    @variable(model, vz_s[1:n_samples, 1:n])

    for i in 1:n_samples, j in 1:n
        set_lower_bound(x_s[i, j], p.x_start - x_margin)
        set_upper_bound(x_s[i, j], p.x_stop + x_margin)
        set_lower_bound(y_s[i, j], -x_margin)
        set_upper_bound(y_s[i, j], x_margin)
        set_lower_bound(z_s[i, j], -bounds.z_margin)
        set_upper_bound(z_s[i, j], bounds.z_margin)
        set_lower_bound(vx_s[i, j], -bounds.v_xy_max)
        set_upper_bound(vx_s[i, j], bounds.v_xy_max)
        set_lower_bound(vy_s[i, j], -bounds.v_xy_max)
        set_upper_bound(vy_s[i, j], bounds.v_xy_max)
        set_lower_bound(vz_s[i, j], -bounds.v_z_max)
        set_upper_bound(vz_s[i, j], bounds.v_z_max)
    end

    if guess !== nothing
        set_start_value(T, sum(guess.dt))
        set_start_value(delta_s0, 0.0)
        s_g = abs(ex) > 1e-12 ? (guess.ux .- p.x_start) ./ ex :
              (guess.uy .- p.y_start) ./ ey
        pert_s = clamp.(s_g, 0.0, L) .- L .* τ
        c_s = Φ_s \ pert_s
        for k in 1:Ns
            ;
            set_start_value(coeff_s[k], c_s[k]);
        end
        if msa.ua_mode === :square
            set_start_value(ua_amp, clamp(sum(guess.ua) / length(guess.ua), bounds.ua_min, bounds.ua_max))
        else
            c_ua = Φ_ua \ clamp.(guess.ua, bounds.ua_min, bounds.ua_max)
            for k in 1:Nua
                ;
                set_start_value(coeff_ua[k], c_ua[k]);
            end
        end
    else
        set_start_value(T, 0.02 * p.maxT)
        set_start_value(delta_s0, 0.0)
        for k in 1:Ns
            ;
            set_start_value(coeff_s[k], 0.0);
        end
        if msa.ua_mode === :square
            set_start_value(ua_amp, 0.5 * (bounds.ua_min + bounds.ua_max))
        else
            for k in 1:Nua
                ;
                set_start_value(coeff_ua[k], 0.0);
            end
        end
    end

    for i in 1:n_samples, j in 1:n
        frac = (j-1)/(n-1)
        set_start_value(x_s[i, j], samples[i].x + frac*(p.x_stop - samples[i].x))
        set_start_value(y_s[i, j], samples[i].y + frac*(p.y_stop - samples[i].y))
        set_start_value(z_s[i, j], samples[i].z * (1 - frac))
        set_start_value(vx_s[i, j], samples[i].vx * (1 - frac))
        set_start_value(vy_s[i, j], samples[i].vy * (1 - frac))
        set_start_value(vz_s[i, j], samples[i].vz * (1 - frac))
    end

    x_margin_final = bounds.r_margin_w * w
    for i in 1:n_samples
        @constraint(model, x_s[i, 1] == samples[i].x)
        @constraint(model, y_s[i, 1] == samples[i].y)
        @constraint(model, z_s[i, 1] == samples[i].z)
        @constraint(model, vx_s[i, 1] == samples[i].vx)
        @constraint(model, vy_s[i, 1] == samples[i].vy)
        @constraint(model, vz_s[i, 1] == samples[i].vz)
        @constraint(model, p.x_stop - x_margin_final <= x_s[i, n] <= p.x_stop + x_margin_final)
        @constraint(model, p.y_stop - x_margin_final <= y_s[i, n] <= p.y_stop + x_margin_final)
    end

    @expression(model, Xi_s[i=1:n_samples, j=1:n], z_s[i, j] - cz)

    @expression(model, r1sq_s[i=1:n_samples, j=1:n],
        (x_s[i, j]-p.x_start)^2 + y_s[i, j]^2)
    @expression(model, wXi1_s[i=1:n_samples, j=1:n],
        w2*(1.0+(Xi_s[i, j]/zR)^2))
    @expression(model, f1_s[i=1:n_samples, j=1:n],
        (w2/wXi1_s[i, j])*exp(-2.0*r1sq_s[i, j]/wXi1_s[i, j]))

    @expression(model, r2sq_s[i=1:n_samples, j=1:n],
        (x_s[i, j]-p.x_stop)^2 + (y_s[i, j]-p.y_stop)^2)
    @expression(model, wXi2_s[i=1:n_samples, j=1:n],
        w2*(1.0+(Xi_s[i, j]/zR)^2))
    @expression(model, f2_s[i=1:n_samples, j=1:n],
        (w2/wXi2_s[i, j])*exp(-2.0*r2sq_s[i, j]/wXi2_s[i, j]))

    @expression(model, rasq_s[i=1:n_samples, j=1:n],
        (x_s[i, j]-ux[j])^2 + (y_s[i, j]-uy[j])^2)
    @expression(model, wXia_s[i=1:n_samples, j=1:n],
        wa2*(1.0+(Xi_s[i, j]/zR_a)^2))
    @expression(model, fa_s[i=1:n_samples, j=1:n],
        (wa2/wXia_s[i, j])*exp(-2.0*rasq_s[i, j]/wXia_s[i, j]))

    @expression(model, Fx_s[i=1:n_samples, j=1:n],
        -4.0*p.U0_static*(x_s[i, j]-p.x_start)/wXi1_s[i, j]*f1_s[i, j]
        -
        4.0*p.U0_static*(x_s[i, j]-p.x_stop) / wXi2_s[i, j]*f2_s[i, j]
        -
        4.0*ua[j]*p.U0_aux_max*(x_s[i, j]-ux[j])/wXia_s[i, j]*fa_s[i, j]
    )
    @expression(model, Fy_beam_s[i=1:n_samples, j=1:n],
        -4.0*p.U0_static*y_s[i, j]/(wXi1_s[i, j])*f1_s[i, j]
        -
        4.0*p.U0_static*(y_s[i, j]-p.y_stop)/wXi2_s[i, j]*f2_s[i, j]
        -
        4.0*ua[j]*p.U0_aux_max*(y_s[i, j]-uy[j])/wXia_s[i, j]*fa_s[i, j]
    )
    @expression(model, Fy_s[i=1:n_samples, j=1:n], Fy_beam_s[i, j] - g)
    @expression(model, Fz_s[i=1:n_samples, j=1:n],
        p.U0_static * f1_s[i, j] * (w2/wXi1_s[i, j]) * (Xi_s[i, j]/zR^2) * (4.0*r1sq_s[i, j]/wXi1_s[i, j]-2.0)
        + p.U0_static * f2_s[i, j] * (w2/wXi2_s[i, j]) * (Xi_s[i, j]/zR^2) * (4.0*r2sq_s[i, j]/wXi2_s[i, j]-2.0)
        + ua[j]*p.U0_aux_max*fa_s[i, j]*(wa2/wXia_s[i, j])*(Xi_s[i, j]/zR_a^2)*(4.0*rasq_s[i, j]/wXia_s[i, j]-2.0)
    )

    for i in 1:n_samples, j in 1:(n-1)
        @constraint(model, x_s[i, j+1] - x_s[i, j] == vx_s[i, j]*dt + 0.5*Fx_s[i, j]*dt^2)
        @constraint(model, y_s[i, j+1] - y_s[i, j] == vy_s[i, j]*dt + 0.5*Fy_s[i, j]*dt^2)
        @constraint(model, z_s[i, j+1] - z_s[i, j] == vz_s[i, j]*dt + 0.5*Fz_s[i, j]*dt^2)
        @constraint(model, vx_s[i, j+1] - vx_s[i, j] == 0.5*dt*(Fx_s[i, j] + Fx_s[i, j+1]))
        @constraint(model, vy_s[i, j+1] - vy_s[i, j] == 0.5*dt*(Fy_s[i, j] + Fy_s[i, j+1]))
        @constraint(model, vz_s[i, j+1] - vz_s[i, j] == 0.5*dt*(Fz_s[i, j] + Fz_s[i, j+1]))
    end

    v_s_max = bounds.v_u_max_per_w * w
    v_ua_max = bounds.v_ua_max
    for j in 1:(n-1)
        @constraint(model, s[j+1]-s[j] <= v_s_max * dt)
        @constraint(model, s[j]-s[j+1] <= v_s_max * dt)
        if msa.ua_mode !== :square
            @constraint(model, ua[j+1]-ua[j] <= v_ua_max * dt)
            @constraint(model, ua[j]-ua[j+1] <= v_ua_max * dt)
        end
    end

    @expression(model, U_st1_s[i=1:n_samples, j=1:n], -p.U0_static * f1_s[i, j])
    @expression(model, U_st2_s[i=1:n_samples, j=1:n], -p.U0_static * f2_s[i, j])
    @expression(model, U_aux_s[i=1:n_samples, j=1:n], -ua[j] * p.U0_aux_max * fa_s[i, j])
    @expression(model, U_tot_s[i=1:n_samples, j=1:n], U_st1_s[i, j]+U_st2_s[i, j]+U_aux_s[i, j])
    @expression(model, KE_s[i=1:n_samples, j=1:n],
        0.5*(vx_s[i, j]^2 + vy_s[i, j]^2 + vz_s[i, j]^2))
    @expression(model, E_tot_s[i=1:n_samples, j=1:n], KE_s[i, j] + U_tot_s[i, j] + g*y_s[i, j])
    @constraint(model, [i=1:n_samples], E_tot_s[i, 1] <= p.starting_trap_fraction * U_tot_s[i, 1])
    @constraint(model, [i=1:n_samples], E_tot_s[i, n] <= p.final_trap_fraction * U_tot_s[i, n])
    if p.trap_fraction !== nothing
        @constraint(model, [i=1:n_samples, j=2:(n-1)], E_tot_s[i, j] <= p.trap_fraction * U_tot_s[i, j])
    end

    println("λ_heat=$(p.lambda_heat)  λ_slosh=$(msa.lambda_slosh)  λ_mse=$(msa.lambda_mse)  use_primary=$(msa.use_primary)  use_secondary=$(msa.use_secondary)")

    @expression(model, heat_s[i=1:n_samples],
        sum(-(U_st1_s[i, j]+U_st2_s[i, j]+U_aux_s[i, j]) for j=1:n))

    @expression(model, slosh_term,
        (sum(x_s[i, n] for i=1:n_samples)/n_samples - p.x_stop)^2 +
        (sum(y_s[i, n] for i=1:n_samples)/n_samples - p.y_stop)^2 +
        (sum(vx_s[i, n] for i=1:n_samples)/n_samples)^2 +
        (sum(vy_s[i, n] for i=1:n_samples)/n_samples)^2
    )
    @expression(model, mse_term,
        sum((x_s[i, n]-p.x_stop)^2 + (y_s[i, n]-p.y_stop)^2 +
            vx_s[i, n]^2 + vy_s[i, n]^2 for i=1:n_samples) / n_samples
    )

    lambda_slosh = msa.use_primary ? msa.lambda_slosh : 0.0
    lambda_mse = msa.use_secondary ? msa.lambda_mse : 0.0

    @objective(model, Min,
        T
        + lambda_slosh * slosh_term
        + lambda_mse * mse_term
    )

    ua_desc = msa.ua_mode === :square ? "square pulse" : "N_modes_ua=$(msa.N_modes_ua)"
    println("Starting 3D MSA optimization with $n_samples samples (N_modes_s=$(msa.N_modes_s), ua_mode=$(msa.ua_mode) [$ua_desc])...")
    optimize!(model)
    println("Termination status (3D MSA): ", termination_status(model))

    t_grid = collect(0.0:1.0:(n-1)) .* value(T) / (n - 1)
    s_val = value.(s)
    ux_val = p.x_start .+ s_val .* ex
    uy_val = p.y_start .+ s_val .* ey

    x_avg = [sum(value(x_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]
    y_avg = [sum(value(y_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]
    z_avg = [sum(value(z_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]
    vx_avg = [sum(value(vx_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]
    vy_avg = [sum(value(vy_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]
    vz_avg = [sum(value(vz_s[i, j]) for i=1:n_samples)/n_samples for j=1:n]

    protocol = ControlProtocol3D(
        t_grid, x_avg, y_avg, z_avg, vx_avg, vy_avg, vz_avg,
        ux_val, uy_val, value.(ua),
    )

    x_traj = zeros(n_samples, n)
    y_traj = zeros(n_samples, n)
    z_traj = zeros(n_samples, n)
    vx_traj = zeros(n_samples, n)
    vy_traj = zeros(n_samples, n)
    vz_traj = zeros(n_samples, n)
    for i in 1:n_samples, j in 1:n
        x_traj[i, j] = value(x_s[i, j])
        y_traj[i, j] = value(y_s[i, j])
        z_traj[i, j] = value(z_s[i, j])
        vx_traj[i, j] = value(vx_s[i, j])
        vy_traj[i, j] = value(vy_s[i, j])
        vz_traj[i, j] = value(vz_s[i, j])
    end

    coeff_ua_val = msa.ua_mode === :square ? [value(ua_amp)] : value.(coeff_ua)

    return MSAControlResult3D(
        protocol,
        samples,
        x_traj, y_traj, z_traj,
        vx_traj, vy_traj, vz_traj,
        value.(coeff_s), coeff_ua_val,
        value(delta_s0),
        seed,
        n_samples,
        string(termination_status(model)),
        objective_value(model),
    )
end

end
