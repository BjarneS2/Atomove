using atomove
using HDF5
using Dates

consts = default_constants3d()

kB = consts.kB
m = consts.m
w0_SI = consts.w0_um * 1e-6
t0_SI = consts.t0_us * 1e-6
v0 = w0_SI / t0_SI
E0 = m * v0^2

velocity_to_dimless(v_mps) = v_mps / v0

function check_transport_feasibility(v_xy_max_mps, dist_um, maxT_us)
    v_dimless = velocity_to_dimless(v_xy_max_mps)
    t_min_us = dist_um / v_dimless
    if t_min_us > maxT_us
        @warn "Transport infeasible: $(dist_um) μm at ≤ $(v_xy_max_mps*1e3) mm/s requires ≥ $(round(t_min_us; digits=1)) μs, but maxT = $(maxT_us) μs."
    else
        println("Feasibility OK: $(dist_um) μm at ≤ $(v_xy_max_mps*1e3) mm/s needs ≥ $(round(t_min_us; digits=1)) μs (maxT = $(maxT_us) μs).")
    end
    return v_dimless
end

T_tweezer = 287e-6
T_atom = 20e-6
println("T_tweezer = $(T_tweezer*1e6) μK, T_atom = $(T_atom*1e6) μK")

U0_static = kB * T_tweezer / E0
U0_aux_max = 3.0 * U0_static

lambda_static_nm = consts.wavelength_static_nm
lambda_dynamic_nm = consts.wavelength_dynamic_nm
w_static_um = 1.17
w_dynamic_um = 1.17
zR_static = π * w_static_um^2 / (lambda_static_nm * 1e-3)
zR_dynamic = π * w_dynamic_um^2 / (lambda_dynamic_nm * 1e-3)

dist_um = 4.6
x_stop = dist_um
y_stop = 0.0

params = TweezerParams3D(
    w=w_static_um,
    w_aux_factor=w_dynamic_um / w_static_um,
    zR=zR_static,
    zR_aux=zR_dynamic,
    x_start=0.0,
    y_start=0.0,
    x_stop=x_stop,
    y_stop=y_stop,
    n=1001,
    maxT=500.0,
    U0_static=U0_static,
    U0_aux_max=U0_aux_max,
    T_tweezer=T_tweezer,
    T_atom=T_atom,
    lambda_heat=0.0,
    lambda_jitter_pos=5.0,
    lambda_jitter_ua=0.5,
    starting_trap_fraction=0.8,
    trap_fraction=0.0,
    final_trap_fraction=0.2,
)

v_xy_max_mps = 0.4      # m/s
v_xy_max = check_transport_feasibility(v_xy_max_mps, dist_um, params.maxT)

bounds = ControlBounds3D(
    T_min_fraction=0.01,
    r_margin_w=0.5,
    z_margin=3.0 * zR_static,
    v_xy_max=v_xy_max,
    v_z_max=0.1,
    u_margin_w=1.0,
    ua_min=0.0,
    ua_max=3.0,
    v_u_max_per_w=3.0,
    v_ua_max=10.0,
)

seed = 37
n_samples = 1

println("U0_static (dimless)  = $U0_static")
println("U0_aux_max (dimless) = $U0_aux_max")
println("zR_static (μm)       = $zR_static")
println("zR_dynamic (μm)      = $zR_dynamic")
println("g_dimless            = $(consts.g_SI * t0_SI^2 / w0_SI)")
println("\nRunning 3D thermal optimal control with $n_samples samples (seed=$seed)...")

file = "C://dev//GitHub//Optimal-Control-of-Atomic-Motion-in-Optical-Tweezer-Arrays//ResultsForThesis//control3d_thermal_2026-07-05_21-03-39.h5"
guess = load_guess_from_file(file, params)
#guess = nothing

result = optimize_controls3d_thermal(
    params;
    guess=guess,
    bounds=bounds,
    n_samples=n_samples,
    seed=seed,
    max_iter=5000,
    hessian_approximation=false,
    print_level=5,
    consts=consts,
    linear_solver="spral",   # "spral" for an apples-to-apples baseline
    fix_final_pos=false
)

println("Objective: $(result.objective_value)  |  Status: $(result.termination_status)")

scales = compute_scales3d_full(params; consts=consts)

results_dir = joinpath(@__DIR__, "..", "results")
isdir(results_dir) || mkdir(results_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = joinpath(results_dir, "control3d_thermal_$(timestamp).h5")
println("Saving to $filename")

proto = result.protocol

ex, ey, L3d = transport_direction(params)
if guess !== nothing
    ig_T = sum(guess.dt)
    ig_ux = copy(guess.ux)
    ig_uy = copy(guess.uy)
    ig_ua = copy(guess.ua)
else
    ig_T = 0.5 * params.maxT
    n3d = params.n
    ig_ux = [params.x_start + (j-1)/(n3d-1) * L3d * ex for j in 1:n3d]
    ig_uy = [params.y_start + (j-1)/(n3d-1) * L3d * ey for j in 1:n3d]
    ig_ua = zeros(n3d)
end

h5open(filename, "w") do file
    attrs(file)["w"] = params.w
    attrs(file)["w_aux_factor"] = params.w_aux_factor
    attrs(file)["zR"] = params.zR
    attrs(file)["zR_aux"] = params.zR_aux
    attrs(file)["x_start"] = params.x_start
    attrs(file)["y_start"] = params.y_start
    attrs(file)["x_stop"] = params.x_stop
    attrs(file)["y_stop"] = params.y_stop
    attrs(file)["n"] = params.n
    attrs(file)["maxT"] = params.maxT
    attrs(file)["U0_static"] = params.U0_static
    attrs(file)["U0_aux_max"] = params.U0_aux_max
    attrs(file)["T_tweezer"] = params.T_tweezer
    attrs(file)["T_atom"] = params.T_atom
    attrs(file)["starting_trap_fraction"] = params.starting_trap_fraction
    attrs(file)["trap_fraction"] = params.trap_fraction === nothing ? -1.0 : params.trap_fraction
    attrs(file)["final_trap_fraction"] = params.final_trap_fraction
    attrs(file)["lambda_heat"] = params.lambda_heat
    attrs(file)["lambda_jitter_pos"] = params.lambda_jitter_pos
    attrs(file)["lambda_jitter_ua"] = params.lambda_jitter_ua
    attrs(file)["move_in_single_trap"] = params.move_in_single_trap
    attrs(file)["single_trap_amplitude"] = params.single_trap_amplitude
    attrs(file)["n_samples"] = result.n_samples
    attrs(file)["seed"] = result.seed === nothing ? -1 : result.seed
    attrs(file)["status"] = result.termination_status
    attrs(file)["objective"] = result.objective_value
    attrs(file)["w0_um"] = consts.w0_um
    attrs(file)["t0_us"] = consts.t0_us
    attrs(file)["v0_m_s"] = scales.v0
    attrs(file)["E0_J"] = scales.E0
    attrs(file)["g_dimless"] = scales.g_dimless
    attrs(file)["v_xy_max"] = v_xy_max

    write(file, "t", proto.t)
    write(file, "x", proto.x)
    write(file, "y", proto.y)
    write(file, "z", proto.z)
    write(file, "vx", proto.vx)
    write(file, "vy", proto.vy)
    write(file, "vz", proto.vz)
    write(file, "ux", proto.ux)
    write(file, "uy", proto.uy)
    write(file, "ua", proto.ua)

    attrs(file)["ig_T"] = ig_T
    write(file, "ig_ux", ig_ux)
    write(file, "ig_uy", ig_uy)
    write(file, "ig_ua", ig_ua)

    write(file, "t_us", proto.t)
    write(file, "x_um", proto.x)
    write(file, "y_um", proto.y)
    write(file, "z_um", proto.z)
    write(file, "ux_um", proto.ux)
    write(file, "uy_um", proto.uy)

    init_x = [s.x for s in result.samples]
    init_y = [s.y for s in result.samples]
    init_z = [s.z for s in result.samples]
    init_vx = [s.vx for s in result.samples]
    init_vy = [s.vy for s in result.samples]
    init_vz = [s.vz for s in result.samples]
    write(file, "init_x", init_x)
    write(file, "init_y", init_y)
    write(file, "init_z", init_z)
    write(file, "init_vx", init_vx)
    write(file, "init_vy", init_vy)
    write(file, "init_vz", init_vz)

    write(file, "x_traj", result.x_traj)
    write(file, "y_traj", result.y_traj)
    write(file, "z_traj", result.z_traj)
    write(file, "vx_traj", result.vx_traj)
    write(file, "vy_traj", result.vy_traj)
    write(file, "vz_traj", result.vz_traj)

    final_energies = zeros(result.n_samples)
    for i in 1:result.n_samples
        xf = result.x_traj[i, end];
        yf = result.y_traj[i, end]
        zf = result.z_traj[i, end]
        vxf = result.vx_traj[i, end];
        vyf = result.vy_traj[i, end];
        vzf = result.vz_traj[i, end]
        Uf = potential3d(xf, yf, zf, proto.ux[end], proto.uy[end], proto.ua[end], params)
        KE = 0.5 * (vxf^2 + vyf^2 + vzf^2)
        final_energies[i] = Uf + KE
    end
    write(file, "final_energies", final_energies)
end
