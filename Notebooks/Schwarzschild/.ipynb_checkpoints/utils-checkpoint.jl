function draw_trajectory(xs, ys; freq=floor(Int, length(xs)/500))
    
    xs′ = []
    ys′ = []
        
    for i in 1:length(xs)
        
        if i%freq == 0
            push!(xs′, xs[i])
            push!(ys′, ys[i])
        end
    
    end
    
    circx = []
    circy = []
    
    for θ in 0:0.1:2π + 0.05
        push!(circx, 2*M*cos(θ))
        push!(circy, 2*M*sin(θ))
    end
    
    # Para cambiar la fuente https://gr-framework.org/fonts.html
    # http://julia.cookbook.tips/doku.php?id=plotattributes
    grafica = plot(legend=false, aspect_ratio=1, xlab=L"x \ (M)", ylab=L"y \ (M)", fontfamily="Times")
    
    grafica = plot!(xs′, ys′)
    grafica = plot!(circx, circy, color="black", fill=(0, :black), lw=2)
    
    return grafica
    
end  

function animate_trajectory(xs, ys; gif_name=nothing, freq=floor(Int, length(xs)/50), fps=20) 
    
    anim = @animate for i in 1:freq:length(xs)
    
        xs_p = [xs[j] for j in 1:i]
        ys_p = [ys[j] for j in 1:i]

        x_f = [xs_p[i]]
        y_f = [ys_p[i]]

        x_min, x_max = minimum(xs), maximum(xs)
        y_min, y_max = minimum(ys), maximum(ys)

        draw_trajectory(xs_p, ys_p, freq=10)
        scatter!(x_f, y_f, color="black", markersize=1)
        plot!(xlims=(x_min, x_max), ylims=(y_min, y_max)) 
        
    end

    if gif_name != nothing
        gif(anim, gif_name, fps=fps) 
    end

end