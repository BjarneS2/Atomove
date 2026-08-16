using atomove
using HDF5
using Dates
using Printf
using Random

const SEED = 101
const DEFAULT_SHOTS = 10000
const DEFAULT_EXTENSION_FACTOR = 2.0
const FINAL_TRAP_FRACTION_OVERRIDE = 0.7

const RESULTS_DIR = joinpath(@__DIR__, "..", "ResultsForThesis")

const BASE_FILES = Dict(
    "A" => joinpath(RESULTS_DIR, "control3d_thermal_2026-06-09_22-34-02.h5"),
    "B" => joinpath(RESULTS_DIR, "control3d_thermal_2026-07-05_21-03-39.h5"),
)

const sim_consts = default_constants3d()

function load_protocol3d_and_params(path::String)
    h5open(path, "r") do file
        atr = attrs(file)
        get_attr(key, default) = haskey(atr, key) ? atr[key] : default

        params = TweezerParams3D(
            w=get_attr("w", 1.5),
            w_aux_factor=get_attr("w_aux_factor", 1.0),
            zR=get_attr("zR", 8.0),
            zR_aux=get_attr("zR_aux", 6.0),
            x_start=get_attr("x_start", 0.0),
            y_start=get_attr("y_start", 0.0),
            x_stop=get_attr("x_stop", 4.6),
            y_stop=get_attr("y_stop", 0.0),
            n=get_attr("n", 201),
            maxT=get_attr("maxT", 500.0),
            U0_static=get_attr("U0_static", 0.01),
            U0_aux_max=get_attr("U0_aux_max", 0.03),
            T_tweezer=get_attr("T_tweezer", 287e-6),
            T_atom=get_attr("T_atom", 40e-6),
            starting_trap_fraction=get_attr("starting_trap_fraction", 0.5),
            trap_fraction=get_attr("trap_fraction", 0.5),
            final_trap_fraction=get_attr("final_trap_fraction", 0.5),
            lambda_heat=get_attr("lambda_heat", 0.0),
            lambda_jitter_pos=get_attr("lambda_jitter_pos", get_attr("lambda_jitter", 0.0)),
            lambda_jitter_ua=get_attr("lambda_jitter_ua", 0.0),
            move_in_single_trap=get_attr("move_in_single_trap", false),
            single_trap_amplitude=get_attr("single_trap_amplitude", 1.0),
        )

        ctrl = ControlProtocol3D(
            read(file["t"]),
            read(file["x"]),
            read(file["y"]),
            read(file["z"]),
            read(file["vx"]),
            read(file["vy"]),
            read(file["vz"]),
            read(file["ux"]),
            read(file["uy"]),
            read(file["ua"]),
        )
        return ctrl, params
    end
end

function apply_overrides(params::TweezerParams3D, T_atom::Float64)
    return TweezerParams3D(;
        w=params.w,
        w_aux_factor=params.w_aux_factor,
        zR=params.zR,
        zR_aux=params.zR_aux,
        x_start=params.x_start,
        y_start=params.y_start,
        x_stop=params.x_stop,
        y_stop=params.y_stop,
        n=params.n,
        maxT=params.maxT,
        U0_static=params.U0_static,
        U0_aux_max=params.U0_aux_max,
        T_atom=T_atom,
        T_tweezer=params.T_tweezer,
        starting_trap_fraction=params.starting_trap_fraction,
        trap_fraction=params.trap_fraction,
        final_trap_fraction=FINAL_TRAP_FRACTION_OVERRIDE,
        lambda_heat=params.lambda_heat,
        lambda_jitter_pos=params.lambda_jitter_pos,
        lambda_jitter_ua=params.lambda_jitter_ua,
        max_sigma_position=params.max_sigma_position,
        move_in_single_trap=params.move_in_single_trap,
        single_trap_amplitude=params.single_trap_amplitude,
        z_aux_offset=params.z_aux_offset,
    )
end

linear_shape(tau::Real) = tau
minjerk_shape(tau::Real) = 10.0 * tau^3 - 15.0 * tau^4 + 6.0 * tau^5

function naive_protocol(ctrl::ControlProtocol3D, params::TweezerParams3D, traj_type::String)
    use_offsets = endswith(traj_type, "_offset")
    base_type = use_offsets ? traj_type[1:end-length("_offset")] : traj_type

    shape = base_type == "linear" ? linear_shape :
            base_type == "minjerk" ? minjerk_shape :
            error("Unknown traj_type $traj_type (expected \"linear\", \"minjerk\", \"linear_offset\", or \"minjerk_offset\")")

    t0 = ctrl.t[1]
    T = ctrl.t[end] - t0
    tau = clamp.((ctrl.t .- t0) ./ T, 0.0, 1.0)
    s = shape.(tau)

    x0, x1 = use_offsets ? (ctrl.ux[1], ctrl.ux[end]) : (params.x_start, params.x_stop)
    y0, y1 = use_offsets ? (ctrl.uy[1], ctrl.uy[end]) : (params.y_start, params.y_stop)

    ux = x0 .+ (x1 - x0) .* s
    uy = y0 .+ (y1 - y0) .* s

    return ControlProtocol3D(
        ctrl.t, ctrl.x, ctrl.y, ctrl.z, ctrl.vx, ctrl.vy, ctrl.vz,
        ux, uy, ctrl.ua,
    ), T
end

function extend_protocol3d(protocol::ControlProtocol3D, factor::Real)
    factor <= 0.0 && return protocol
    t_total = protocol.t[end] - protocol.t[1]
    t_ext = factor * t_total
    n_orig = length(protocol.t)
    avg_dt = t_total / (n_orig - 1)
    n_new = max(2, round(Int, t_ext / avg_dt))
    t_tail = range(protocol.t[end], protocol.t[end] + t_ext; length=n_new + 1)[2:end]
    return ControlProtocol3D(
        vcat(protocol.t, collect(t_tail)),
        vcat(protocol.x, fill(protocol.x[end], n_new)),
        vcat(protocol.y, fill(protocol.y[end], n_new)),
        vcat(protocol.z, fill(protocol.z[end], n_new)),
        vcat(protocol.vx, fill(0.0, n_new)),
        vcat(protocol.vy, fill(0.0, n_new)),
        vcat(protocol.vz, fill(0.0, n_new)),
        vcat(protocol.ux, fill(protocol.ux[end], n_new)),
        vcat(protocol.uy, fill(protocol.uy[end], n_new)),
        vcat(protocol.ua, zeros(Float64, n_new)),
    )
end

function main()
    length(ARGS) >= 3 || error("Usage: run_forward_3d_naive_trajectories.jl <A|B> <linear|minjerk> <T_atom_uK> [shots] [extension_factor]")
    base = ARGS[1]
    traj_type = ARGS[2]
    T_atom_uK = parse(Float64, ARGS[3])
    shots = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : DEFAULT_SHOTS
    extension_factor = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_EXTENSION_FACTOR

    haskey(BASE_FILES, base) || error("Unknown base \"$base\" (expected \"A\" or \"B\")")
    base_file = BASE_FILES[base]

    Random.seed!(SEED)

    println("Loading base protocol from $base_file")
    ctrl0, params0 = load_protocol3d_and_params(base_file)
    params = apply_overrides(params0, T_atom_uK * 1e-6)

    protocol, T_transport = naive_protocol(ctrl0, params, traj_type)
    println(@sprintf("Built %s trajectory over base %s (T = %.4f us)", traj_type, base, T_transport))

    if extension_factor > 0.0
        println(@sprintf("Extending by %.2fx transport time", extension_factor))
        protocol = extend_protocol3d(protocol, extension_factor)
    end

    scales = compute_scales3d_full(params; consts=sim_consts)
    n = length(protocol.t)

    x_all = zeros(n, shots);
    y_all = zeros(n, shots);
    z_all = zeros(n, shots)
    vx_all = zeros(n, shots);
    vy_all = zeros(n, shots);
    vz_all = zeros(n, shots)
    lost = falses(shots)

    println(@sprintf("Running %d forward shots...", shots))
    for s in 1:shots
        traj = simulate_forward3d(protocol, params; thermal_sample=true, consts=sim_consts)
        x_all[:, s] = traj.x;
        y_all[:, s] = traj.y;
        z_all[:, s] = traj.z
        vx_all[:, s] = traj.vx;
        vy_all[:, s] = traj.vy;
        vz_all[:, s] = traj.vz
        lost[s] = traj.lost
    end

    survival = 1.0 - sum(lost) / shots
    println(@sprintf("Survival rate: %.3f  (%d/%d)", survival, shots - sum(lost), shots))

    isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = joinpath(RESULTS_DIR, "forward3d_naive_$(traj_type)_$(base)_$(Int(round(T_atom_uK)))uK_$(timestamp).h5")
    println("Saving to $filename")

    h5open(filename, "w") do file
        attrs(file)["protocol_file"] = base_file
        attrs(file)["traj_type"] = traj_type
        attrs(file)["base_run"] = base
        attrs(file)["T_transport_us"] = T_transport
        attrs(file)["timestamp"] = timestamp
        attrs(file)["shots"] = shots
        attrs(file)["extension_factor"] = extension_factor
        attrs(file)["SEED"] = SEED
        attrs(file)["survival_rate"] = survival
        attrs(file)["w0_um"] = sim_consts.w0_um
        attrs(file)["t0_us"] = sim_consts.t0_us
        attrs(file)["v0_m_s"] = scales.v0
        attrs(file)["g_dimless"] = scales.g_dimless
        attrs(file)["w"] = params.w
        attrs(file)["w_aux_factor"] = params.w_aux_factor
        attrs(file)["zR"] = params.zR
        attrs(file)["zR_aux"] = params.zR_aux
        attrs(file)["x_start"] = params.x_start
        attrs(file)["y_start"] = params.y_start
        attrs(file)["x_stop"] = params.x_stop
        attrs(file)["y_stop"] = params.y_stop
        attrs(file)["U0_static"] = params.U0_static
        attrs(file)["U0_aux_max"] = params.U0_aux_max
        attrs(file)["starting_trap_fraction"] = params.starting_trap_fraction
        attrs(file)["trap_fraction"] = params.trap_fraction === nothing ? NaN : params.trap_fraction
        attrs(file)["final_trap_fraction"] = params.final_trap_fraction
        attrs(file)["T_atom"] = params.T_atom
        attrs(file)["T_tweezer"] = params.T_tweezer

        write(file, "t", protocol.t)
        write(file, "t_us", protocol.t)
        write(file, "ux", protocol.ux)
        write(file, "uy", protocol.uy)
        write(file, "ua", protocol.ua)

        write(file, "x", x_all)
        write(file, "y", y_all)
        write(file, "z", z_all)
        write(file, "vx", vx_all)
        write(file, "vy", vy_all)
        write(file, "vz", vz_all)
        write(file, "x_um", x_all)
        write(file, "y_um", y_all)
        write(file, "z_um", z_all)
        write(file, "lost", collect(lost))
    end
end

main()
