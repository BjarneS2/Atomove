module Models3D

export potential3d, forces3d, barrier_height3d

using ..Types3D

@inline function _beam_factor(x::Real, y::Real, z::Real,
    cx::Real, cy::Real, cz::Real,
    w::Float64, zR::Float64)
    Xi = z - cz
    rho2 = (x - cx)^2 + (y - cy)^2
    wXi2 = w^2 * (1.0 + (Xi / zR)^2)
    f = (w^2 / wXi2) * exp(-2.0 * rho2 / wXi2)
    return f, rho2, Xi, wXi2
end

function potential3d(x::Real, y::Real, z::Real,
    ux::Real, uy::Real, ua::Real,
    p::TweezerParams3D)
    w = p.w
    w_aux = p.w * p.w_aux_factor
    zR = p.zR
    zR_a = p.zR_aux
    cz = 0.0
    cz_a = p.z_aux_offset

    f_st1, _, _, _ = _beam_factor(x, y, z, p.x_start, 0.0, cz, w, zR)
    f_st2, _, _, _ = _beam_factor(x, y, z, p.x_stop, p.y_stop, cz, w, zR)
    f_aux, _, _, _ = _beam_factor(x, y, z, ux, uy, cz_a, w_aux, zR_a)

    return -p.U0_static * f_st1 - p.U0_static * f_st2 - ua * p.U0_aux_max * f_aux
end

function potential3d(x::Real, y::Real, z::Real,
    ux::Real, uy::Real, ua::Real,
    p::TweezerParams3D, ::Val{:jump})
    w = p.w
    w_aux = p.w * p.w_aux_factor
    zR = p.zR
    zR_a = p.zR_aux
    cz = 0.0
    cz_a = p.z_aux_offset

    Xi1 = z - cz
    r1sq = (x - p.x_start)^2 + (y - 0.0)^2
    wXi1 = w^2 * (1.0 + (Xi1/zR)^2)
    f1 = (w^2/wXi1)*exp(-2.0*r1sq/wXi1)

    Xi2 = z - cz
    r2sq = (x - p.x_stop)^2 + (y - p.y_stop)^2
    wXi2 = w^2 * (1.0 + (Xi2/zR)^2)
    f2 = (w^2/wXi2)*exp(-2.0*r2sq/wXi2)

    Xia = z - cz_a
    rasq = (x - ux)^2 + (y - uy)^2
    wXia = w_aux^2 * (1.0 + (Xia/zR_a)^2)
    fa = (w_aux^2/wXia)*exp(-2.0*rasq/wXia)

    return -p.U0_static*f1 - p.U0_static*f2 - ua*p.U0_aux_max*fa
end

@inline function _beam_forces(x::Real, y::Real, z::Real,
    cx::Real, cy::Real, cz::Real,
    U0::Real, w::Float64, zR::Float64)
    Xi = z - cz
    dx = x - cx
    dy = y - cy
    rho2 = dx^2 + dy^2
    wXi2 = w^2 * (1.0 + (Xi / zR)^2)
    alpha = w^2 / wXi2
    f = alpha * exp(-2.0 * rho2 / wXi2)

    Fx = -4.0 * U0 * dx / wXi2 * f
    Fy = -4.0 * U0 * dy / wXi2 * f
    Fz = U0 * f * alpha * (Xi / zR^2) * (4.0 * rho2 / wXi2 - 2.0)
    return Fx, Fy, Fz
end

function forces3d(x::Real, y::Real, z::Real,
    ux::Real, uy::Real, ua::Real,
    p::TweezerParams3D, g_dimless::Float64)
    w = p.w
    w_aux = p.w * p.w_aux_factor
    zR = p.zR
    zR_a = p.zR_aux
    cz = 0.0
    cz_a = p.z_aux_offset

    Fx1, Fy1, Fz1 = _beam_forces(x, y, z, p.x_start, 0.0, cz, p.U0_static, w, zR)
    Fx2, Fy2, Fz2 = _beam_forces(x, y, z, p.x_stop, p.y_stop, cz, p.U0_static, w, zR)
    Fxa, Fya, Fza = _beam_forces(x, y, z, ux, uy, cz_a, ua*p.U0_aux_max, w_aux, zR_a)

    Fx = Fx1 + Fx2 + Fxa
    Fy = Fy1 + Fy2 + Fya - g_dimless
    Fz = Fz1 + Fz2 + Fza

    return Fx, Fy, Fz
end

end
