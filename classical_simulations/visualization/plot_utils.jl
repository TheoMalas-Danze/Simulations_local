using Plots

function plot_gif(beg,step,ending)
    @gif for index in 1:length(cat_phase_list)
    alpha_round = round(cat_phase_list[index]/2, digits = 1)
    f_beg, f_end= round(ωk_chain[beg]/2pi,digits=2), round(ωk_chain[ending]/2pi,digits=2)
    x=ωk_chain[beg:step:ending]/2pi
    y=sin.(Phase_centered[index][beg:step:ending])
    z=cos.(Phase_centered[index][beg:step:ending])
    plot3d(
    x, y, z,
    ylim=(-1.1 ,1.1),
    zlim=(-1.1 ,1.1),
    label="cat phase = $(alpha_round)",
    legend=:topright,
    marker=:circle,
    markersize=4,
    xlabel=L"\omega_k/2\pi"*" (GHz)",
    title="[$f_beg GHz, $f_end GHz]",
    )
    end every 1  # you can adjust the frame rate
end

function plot_mode(ϕ::Vector{Float64}, q::Vector{Float64}, t_list::Vector{Float64}, ω::Float64, Z::Float64; phase=0.0)
    Re = ϕ / sqrt(2*Z)
    Im = q * sqrt(Z/2)

    ϕ_rotating =@. ϕ * cos(ω * t_list + phase) + q*Z * sin(-ω * t_list - phase)
    q_rotating =@. -ϕ/Z * sin(-ω * t_list - phase) + q * cos(ω * t_list + phase)

    Re_rotating = ϕ_rotating / sqrt(2*Z)
    Im_rotating = q_rotating * sqrt(Z/2)

    p1=plot(Re,Im)
    p2=plot(Re_rotating,Im_rotating)
    return(p1,p2)
end


function plot_perso(args...; layout=(1,1), kwargs...)
    size = (layout[2], layout[1]) .* size_px_car
    top_margin = (layout[2]-1)*5Plots.mm
    bottom_margin = top_margin
    left_margin = layout[2]*2Plots.mm+(layout[1]-1)*5Plots.mm
    right_margin = left_margin
    plot(args...; size=size, left_margin=left_margin, right_margin=right_margin, top_margin=top_margin, bottom_margin=bottom_margin, layout=layout, kwargs...)
end