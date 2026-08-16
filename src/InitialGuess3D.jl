module InitialGuess3D

using HDF5
using ..Types3D
using ..ForwardDynamics3D

export linear_sweep_guess, sta_guess, load_guess_from_file, load_params_from_file
export propagate_guess_states3d

function _zero_state_guess(p::TweezerParams3D, T::Float64,
                            s_profile::AbstractVector{Float64})
    n  = p.n
    ex, ey, _ = transport_direction(p)
    dt_vec = fill(T / (n - 1), n - 1)

    ux = p.x_start .+ s_profile .* ex
    uy = p.y_start .+ s_profile .* ey

    return InitialG3D(
        dt_vec,
        copy(ux), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(n),
        ux, uy, zeros(n),
    )
end

function linear_sweep_guess(p::TweezerParams3D; T::Float64 = p.maxT / 2.0)
    n = p.n
    L = transport_length(p)
    s_profile = [L * (j - 1) / (n - 1) for j in 1:n]
    return _zero_state_guess(p, T, s_profile)
end

function sta_guess(p::TweezerParams3D; T::Float64 = p.maxT / 2.0)
    n = p.n
    L = transport_length(p)

    s_profile = Vector{Float64}(undef, n)
    for j in 1:n
        τ = (j - 1) / (n - 1)
        s_profile[j] = L * τ^3 * (10.0 - 15.0 * τ + 6.0 * τ^2)
    end

    return _zero_state_guess(p, T, s_profile)
end

function load_guess_from_file(path::AbstractString, p::TweezerParams3D)
    t_src, x_src, y_src, z_src, vx_src, vy_src, vz_src, ux_src, uy_src, ua_src =
        h5open(path, "r") do f
            if haskey(f, "y")
                read(f, "t"),
                read(f, "x"), read(f, "y"), read(f, "z"),
                read(f, "vx"), read(f, "vy"), read(f, "vz"),
                read(f, "ux"), read(f, "uy"), read(f, "ua")
            else
                t   = read(f, "t")
                x   = read(f, "x")
                v   = read(f, "v")
                ux  = read(f, "ux")
                ua  = read(f, "ua")
                n2  = length(t)
                t, x, zeros(n2), zeros(n2),
                v, zeros(n2), zeros(n2),
                ux, fill(p.y_start, n2), ua
            end
        end

    T_src = t_src[end]
    t_dst = [T_src * (j - 1) / (p.n - 1) for j in 1:p.n]

    interp(src, t_s, t_d) = [_linterp(t_s, src, t) for t in t_d]

    x_dst  = interp(x_src,  t_src, t_dst)
    y_dst  = interp(y_src,  t_src, t_dst)
    z_dst  = interp(z_src,  t_src, t_dst)
    vx_dst = interp(vx_src, t_src, t_dst)
    vy_dst = interp(vy_src, t_src, t_dst)
    vz_dst = interp(vz_src, t_src, t_dst)
    ux_dst = interp(ux_src, t_src, t_dst)
    uy_dst = interp(uy_src, t_src, t_dst)
    ua_dst = interp(ua_src, t_src, t_dst)

    dt_vec = fill(T_src / (p.n - 1), p.n - 1)

    return InitialG3D(
        dt_vec,
        x_dst, y_dst, z_dst,
        vx_dst, vy_dst, vz_dst,
        ux_dst, uy_dst, ua_dst,
    )
end

function _resample_control3d(guess::InitialG3D, n::Int)
    T   = sum(guess.dt)
    n_g = length(guess.ua)
    t_grid = [T * (j - 1) / (n - 1) for j in 1:n]

    if n_g == n
        return t_grid, copy(guess.ux), copy(guess.uy), copy(guess.ua)
    end

    function interp_guess(arr::Vector{Float64}, j::Int)
        τ     = (j - 1) / (n - 1)
        g_idx = τ * (n_g - 1)
        lo    = clamp(floor(Int, g_idx) + 1, 1, n_g)
        hi    = clamp(lo + 1, 1, n_g)
        alpha = g_idx - (lo - 1)
        return (1 - alpha) * arr[lo] + alpha * arr[hi]
    end

    ux_n = [interp_guess(guess.ux, j) for j in 1:n]
    uy_n = [interp_guess(guess.uy, j) for j in 1:n]
    ua_n = [interp_guess(guess.ua, j) for j in 1:n]
    return t_grid, ux_n, uy_n, ua_n
end

function propagate_guess_states3d(guess::InitialG3D, samples, p::TweezerParams3D;
                                   consts::PhysicalConstants3D = default_constants3d())
    n = p.n
    t_grid, ux_n, uy_n, ua_n = _resample_control3d(guess, n)

    ctrl = ControlProtocol3D(
        t_grid, zeros(n), zeros(n), zeros(n),
        zeros(n), zeros(n), zeros(n),
        ux_n, uy_n, ua_n,
    )

    trajs = Vector{Trajectory3D}(undef, length(samples))
    for (i, sample) in enumerate(samples)
        trajs[i] = simulate_forward3d(
            ctrl, p;
            thermal_sample = false,
            consts         = consts,
            init = (x = sample.x, y = sample.y, z = sample.z,
                    vx = sample.vx, vy = sample.vy, vz = sample.vz),
        )
    end

    return ux_n, uy_n, ua_n, trajs
end

function load_params_from_file(path::AbstractString)
    h5open(path, "r") do f
        a = attrs(f)

        tf_raw = read(a["trap_fraction"])
        starting_tf = haskey(a, "starting_trap_fraction") ?
            read(a["starting_trap_fraction"]) : tf_raw
        trap_frac = tf_raw == -1.0 ? nothing : tf_raw

        consts = PhysicalConstants3D(
            w0_um = read(a["w0_um"]),
            t0_us = read(a["t0_us"]),
        )

        params = TweezerParams3D(
            w                      = read(a["w"]),
            w_aux_factor           = read(a["w_aux_factor"]),
            zR                     = read(a["zR"]),
            zR_aux                 = read(a["zR_aux"]),
            x_start                = read(a["x_start"]),
            y_start                = read(a["y_start"]),
            x_stop                 = read(a["x_stop"]),
            y_stop                 = read(a["y_stop"]),
            n                      = read(a["n"]),
            maxT                   = read(a["maxT"]),
            U0_static              = read(a["U0_static"]),
            U0_aux_max             = read(a["U0_aux_max"]),
            T_tweezer              = read(a["T_tweezer"]),
            T_atom                 = read(a["T_atom"]),
            starting_trap_fraction = starting_tf,
            trap_fraction          = trap_frac,
            final_trap_fraction    = read(a["final_trap_fraction"]),
            lambda_heat            = read(a["lambda_heat"]),
            lambda_jitter_pos      = haskey(a, "lambda_jitter_pos") ?
                                         read(a["lambda_jitter_pos"]) :
                                         read(a["lambda_jitter"]),
            lambda_jitter_ua       = haskey(a, "lambda_jitter_ua") ?
                                         read(a["lambda_jitter_ua"]) :
                                         read(a["lambda_jitter"]),
            move_in_single_trap    = haskey(a, "move_in_single_trap") ?
                                         read(a["move_in_single_trap"]) : false,
            single_trap_amplitude  = haskey(a, "single_trap_amplitude") ?
                                         read(a["single_trap_amplitude"]) : 1.0,
        )

        return params, consts
    end
end

function _linterp(t::AbstractVector{Float64}, v::AbstractVector{Float64}, t0::Float64)
    t0 <= t[1]   && return v[1]
    t0 >= t[end] && return v[end]
    lo, hi = 1, length(t)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        t[mid] <= t0 ? (lo = mid) : (hi = mid)
    end
    α = (t0 - t[lo]) / (t[hi] - t[lo])
    return v[lo] + α * (v[hi] - v[lo])
end

end
