using atomove
using HDF5
using Dates
using Printf
using Random

const SEED = 101
const SHOTS = 5000
const EXTENSION_FACTOR = 1.0   # same relative extension used for the 1D benchmark

const SRC_DIR = raw"c:\dev\GitHub\MasterThesisJulia\results\used_for_prep"
const OUT_DIR = SRC_DIR

# MTJ (1D) physical constants — fixed, same for every protocol in that framework
const MTJ_kB = 1.380649e-23
const MTJ_m = 2.273347e-25
const MTJ_w0_um = 1.4            # length scale used for ALL positions in MTJ

const LAMBDA_STATIC_UM = 0.936   # 936 nm
const LAMBDA_DYNAMIC_UM = 0.933  # 933 nm

const sim_consts3d = default_constants3d()

function mtj_t0_us(T_tweezer::Float64)
    w0_SI = MTJ_w0_um * 1e-6
    U0_J = MTJ_kB * T_tweezer
    omega0 = sqrt(4 * U0_J / (MTJ_m * w0_SI^2))
    t0_s = 1 / omega0
    return t0_s * 1e6
end

function load_mtj_protocol(path::String)
    h5open(path, "r") do file
        atr = attrs(file)
        get_attr(key, default) = haskey(atr, key) ? atr[key] : default

        w = get_attr("w", 1.0)
        w_aux_factor = get_attr("w_aux_factor", 1.0)
        xStart = get_attr("xStart", 0.0)
        xStop = get_attr("xStop", 1.0)
        T_tweezer = get_attr("T_tweezer", 287e-6)
        T_atom = get_attr("T_atom", 40e-6)
        trap_fraction = get_attr("trap_fraction", 0.5)

        t = read(file["t"])
        x = read(file["x"])
        ux = read(file["ux"])
        ua = read(file["ua"])

        return (t=t, x=x, ux=ux, ua=ua, w=w, w_aux_factor=w_aux_factor,
                xStart=xStart, xStop=xStop, T_tweezer=T_tweezer, T_atom=T_atom,
                trap_fraction=trap_fraction)
    end
end

function extend_protocol3d(t, ux, uy, ua, extension_factor::Float64)
    extension_factor <= 0.0 && return t, ux, uy, ua
    t_total = t[end] - t[1]
    t_ext = extension_factor * t_total
    n_orig = length(t)
    avg_dt = t_total / (n_orig - 1)
    n_new = max(2, round(Int, t_ext / avg_dt))
    t_tail = collect(range(t[end], t[end] + t_ext; length=n_new + 1)[2:end])
    return (
        vcat(t, t_tail),
        vcat(ux, fill(ux[end], n_new)),
        vcat(uy, fill(uy[end], n_new)),
        vcat(ua, zeros(Float64, n_new)),
    )
end

function build_3d_protocol_and_params(mtj)
    t0_us = mtj_t0_us(mtj.T_tweezer)

    # Physical unit conversion (all positions scaled by the fixed MTJ length unit)
    t_phys_us = mtj.t .* t0_us
    ux_phys_um = mtj.ux .* MTJ_w0_um
    xStop_phys_um = mtj.xStop * MTJ_w0_um

    uy_phys_um = zeros(Float64, length(t_phys_us))

    # 3D atomove dimensionless units: 1 unit = 1 μm, 1 unit = 1 μs → values equal physical μm/μs numerically
    t3d, ux3d, uy3d, ua3d = extend_protocol3d(t_phys_us, ux_phys_um, uy_phys_um, copy(mtj.ua), EXTENSION_FACTOR)

    w_stat_um = mtj.w * MTJ_w0_um
    w_aux_um = w_stat_um * mtj.w_aux_factor

    zR = pi * w_stat_um^2 / LAMBDA_STATIC_UM
    zR_aux = pi * w_aux_um^2 / LAMBDA_DYNAMIC_UM

    # 3D energy unit and dimensionless trap depths
    v0_3d = 1.0   # w0_3d_um / t0_3d_us = 1/1
    E0_3d = sim_consts3d.m * v0_3d^2
    U0_static = MTJ_kB * mtj.T_tweezer / E0_3d
    U0_aux_max = U0_static  # same nominal depth as static trap; ua scales it 0..1, matching the 1D model

    params = TweezerParams3D(
        w=w_stat_um,
        w_aux_factor=mtj.w_aux_factor,
        zR=zR,
        zR_aux=zR_aux,
        x_start=0.0,
        y_start=0.0,
        x_stop=xStop_phys_um,
        y_stop=0.0,
        n=length(t3d),
        maxT=t3d[end],
        U0_static=U0_static,
        U0_aux_max=U0_aux_max,
        T_atom=mtj.T_atom,
        T_tweezer=mtj.T_tweezer,
        starting_trap_fraction=mtj.trap_fraction,
        trap_fraction=mtj.trap_fraction,
        final_trap_fraction=0.5,
    )

    ctrl = ControlProtocol3D(
        t3d,
        zeros(Float64, length(t3d)), zeros(Float64, length(t3d)), zeros(Float64, length(t3d)),
        zeros(Float64, length(t3d)), zeros(Float64, length(t3d)), zeros(Float64, length(t3d)),
        ux3d, uy3d, ua3d,
    )

    return ctrl, params
end

function run_one(protocol_file::String)
    println("=== Converting + running $protocol_file ===")
    mtj = load_mtj_protocol(protocol_file)
    ctrl, params = build_3d_protocol_and_params(mtj)

    n = length(ctrl.t)
    x_all = zeros(n, SHOTS); y_all = zeros(n, SHOTS); z_all = zeros(n, SHOTS)
    vx_all = zeros(n, SHOTS); vy_all = zeros(n, SHOTS); vz_all = zeros(n, SHOTS)
    lost = falses(SHOTS)

    println(@sprintf("  w=%.4f um  w_aux=%.4f um  zR=%.4f  zR_aux=%.4f  x_stop=%.4f um  U0_static=%.5g  trap_fraction=%.3f  T_atom=%.3g K",
        params.w, params.w * params.w_aux_factor, params.zR, params.zR_aux, params.x_stop, params.U0_static, params.trap_fraction, params.T_atom))
    println(@sprintf("  Running %d shots...", SHOTS))

    for s in 1:SHOTS
        traj = simulate_forward3d(ctrl, params; thermal_sample=true, consts=sim_consts3d)
        x_all[:, s] = traj.x; y_all[:, s] = traj.y; z_all[:, s] = traj.z
        vx_all[:, s] = traj.vx; vy_all[:, s] = traj.vy; vz_all[:, s] = traj.vz
        lost[s] = traj.lost
    end

    base = splitext_basename(protocol_file)
    name = replace(base, "control_protocol" => "forward3d")
    filename = joinpath(OUT_DIR, "$(name)_5000shots.h5")
    println("  Saving to $filename")

    h5open(filename, "w") do file
        attrs(file)["protocol_file"] = protocol_file
        attrs(file)["shots"] = SHOTS
        attrs(file)["extension_factor"] = EXTENSION_FACTOR
        attrs(file)["SEED"] = SEED
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
        attrs(file)["trap_fraction"] = params.trap_fraction
        attrs(file)["final_trap_fraction"] = params.final_trap_fraction
        attrs(file)["T_atom"] = params.T_atom
        attrs(file)["T_tweezer"] = params.T_tweezer
        attrs(file)["w0_um"] = sim_consts3d.w0_um
        attrs(file)["t0_us"] = sim_consts3d.t0_us

        write(file, "t", ctrl.t)
        write(file, "ux", ctrl.ux)
        write(file, "uy", ctrl.uy)
        write(file, "ua", ctrl.ua)

        write(file, "x", x_all); write(file, "y", y_all); write(file, "z", z_all)
        write(file, "vx", vx_all); write(file, "vy", vy_all); write(file, "vz", vz_all)
        write(file, "lost", collect(lost))
    end

    println(@sprintf("  -> Julia-internal loss flag: %.2f%% (%d/%d)", 100*sum(lost)/SHOTS, sum(lost), SHOTS))
    return filename
end

splitext_basename(path) = splitext(basename(path))[1]

function main()
    Random.seed!(SEED)
    protocol_files = sort(filter(f -> occursin("control_protocol", basename(f)) && endswith(f, ".h5"), readdir(SRC_DIR; join=true)))
    isempty(protocol_files) && error("No control protocol files found in $SRC_DIR")

    println("Found $(length(protocol_files)) MTJ (1D) control protocol file(s) to benchmark in the 3D model:")
    for f in protocol_files
        println("  $f")
    end

    outputs = String[]
    for f in protocol_files
        push!(outputs, run_one(f))
    end

    println("\n=== Done. 3D forward trajectory files: ===")
    for o in outputs
        println("  $o")
    end
end

main()
