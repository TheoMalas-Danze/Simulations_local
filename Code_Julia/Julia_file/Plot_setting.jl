using Printf, Plots, LaTeXStrings
gr()  # use GR backend

dpi = 300
width_in = 3.54 /1.2
height_in = width_in / 1.6
size_px = (round(Int,width_in * dpi), round(Int,height_in * dpi))
size_px_car = (round(Int,(width_in+height_in)/2 * dpi), round(Int,(width_in+height_in)/2 * dpi))

Perso_palette = [(0, 0, 0), (47, 72, 88), (51, 101, 138), (134, 187, 216), (160, 40, 41), (151, 161, 105)]
Perso_palette_hex = ["#000000", "#2f4858", "#33658a", "#86bbd8", "#a02829", "#97a169"]


linestyles = [:solid, :dash, :dot, :dashdot]

function rgb_to_hex(rgb)
    return @sprintf("#%02x%02x%02x", rgb...)
end

function hex_to_rgb(hex::String)
    parse.(Int, [hex[i:i+1] for i in 2:2:6]; base=16)
end



ORANGE = (255, 165, 0)

function lininterp(c1, c2, f, alpha=1.0)
    out = [(1-f)*c1[i] + f*c2[i] for i in 1:3] ./ 255
    return clamp.(out, 0, 1), alpha
end

function colorMap(f)
    if 0 <= f < 0.5
        return lininterp(Perso_palette[1], Perso_palette[4], 2f)
    elseif 0.5 <= f < 0.75
        return lininterp(Perso_palette[4], ORANGE, 4*(f - 0.5))
    elseif 0.75 <= f <= 1
        return lininterp(ORANGE, Perso_palette[5], 4*(f - 0.75))
    end
end

colors = [RGBA(r, g, b, α) for f in 0.0:0.001:1.0 for (rgb, α) in [colorMap(f)] for (r, g, b) = (rgb,)]
Perso_colormap = cgrad(colors)

chop(arr; max=1e-3) = [abs(x) > max ? x : 0 for x in arr];

default(
    size=size_px,
    dpi=dpi,
    #####################
    margin = 5Plots.mm,
    #####################
    fontfamily="Computer Modern",
    titlefontsize=20,
    guidefontsize=20,
    tickfontsize=20,
    legendfontsize=16,
    colorbar_titlefontsize=20,
    #####################
    grid = true,
    gridlinewidth = 1.5,
    gridcolor = :gray,
    gridalpha = 0.4,
    #####################
    linewidth=2,
    markersize=8,
    #####################
    palette=Perso_palette_hex,
    markerstrokecolor = :auto,
)
mm=1Plots.mm;