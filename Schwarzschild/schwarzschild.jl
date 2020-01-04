function impact_parameter(ϕ0, ϕ_final, u0, dϕ)
    
    ϕs = []
    us = []
    intervalo = []
    
    push!(ϕs, ϕ0)
    push!(us, u0)
    
    f1(ϕ, u, w) = w             # esta es la de u
    f2(ϕ, u, w) = 3*M*u^2 - u   # esta es la de w = du/dϕ
        
    dr_dλ = -sqrt((1-2*M*u0)*(ϵ^2/(1-2*M*u0)-(l*u0)^2))
    dϕ_dλ = l*u0^2
    w0 = -u0^2*(dr_dλ/dϕ_dλ)
    
    ϕ = ϕ0
    u = u0
    w = w0
    
    while ϕ < ϕ_final
        
        k1 = dϕ*f1(ϕ, u, w) # esta es la de u
        j1 = dϕ*f2(ϕ, u, w) # esta es la de w = du/dϕ
        
        k2 = dϕ*f1(ϕ + 0.5*dϕ, u + 0.5*k1, w + 0.5*j1)
        j2 = dϕ*f2(ϕ + 0.5*dϕ, u + 0.5*k1, w + 0.5*j1)
        
        k3 = dϕ*f1(ϕ + 0.5*dϕ, u + 0.5*k2, w + 0.5*j2)
        j3 = dϕ*f2(ϕ + 0.5*dϕ, u + 0.5*k2, w + 0.5*j2) 
        
        k4 = dϕ*f1(ϕ + dϕ, u + k3, w + j3)
        j4 = dϕ*f2(ϕ + dϕ, u + k3, w + j3)
        
        ϕ = ϕ + dϕ
        u = u + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        w = w + (1/6)*(j1 + 2*j2 + 2*j3 + j4)
                        
        push!(ϕs, ϕ)
        push!(us, u)  
               
    end
    
    rs = []
    
    for i in 1:length(us)
        push!(rs, 1/us[i])
    end
    
    xs = []
    ys = []
    
    for i in 1:length(rs)
        push!(xs, rs[i]*cos(ϕs[i]))
        push!(ys, rs[i]*sin(ϕs[i]))
    end
    
    return xs, ys
    
end

function impact_parameter2(ϕ0, ϕ_final, r0, dϕ)
    
    ϕs = []
    rs = []
    intervalo = []
    
    push!(ϕs, ϕ0)
    push!(rs, r0)
    
    f1(ϕ, r, w) = w                   # esta es la de r
    f2(ϕ, r, w) = 2*r^3/b^2 - r + M   # esta es la de w = dr/dϕ
        
    dr_dλ = -sqrt((1-2*M/r0)*(ϵ^2/(1-2*M/r0)-(l/r0)^2))
    dϕ_dλ = l/r0^2
    w0 = (dr_dλ/dϕ_dλ)
    
    ϕ = ϕ0
    r = r0
    w = w0
    
    while ϕ < ϕ_final
        
        k1 = dϕ*f1(ϕ, r, w) # esta es la de u
        j1 = dϕ*f2(ϕ, r, w) # esta es la de w = du/dϕ
        
        k2 = dϕ*f1(ϕ + 0.5*dϕ, r + 0.5*k1, w + 0.5*j1)
        j2 = dϕ*f2(ϕ + 0.5*dϕ, r + 0.5*k1, w + 0.5*j1)
        
        k3 = dϕ*f1(ϕ + 0.5*dϕ, r + 0.5*k2, w + 0.5*j2)
        j3 = dϕ*f2(ϕ + 0.5*dϕ, r + 0.5*k2, w + 0.5*j2) 
        
        k4 = dϕ*f1(ϕ + dϕ, r + k3, w + j3)
        j4 = dϕ*f2(ϕ + dϕ, r + k3, w + j3)
        
        ϕ = ϕ + dϕ
        r = r + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        w = w + (1/6)*(j1 + 2*j2 + 2*j3 + j4)
                        
        push!(ϕs, ϕ)
        push!(rs, r)  
               
    end
    
    xs = []
    ys = []
        
    for i in 1:length(rs)
        push!(xs, rs[i]*cos(ϕs[i]))
        push!(ys, rs[i]*sin(ϕs[i]))
    end
    
    return xs, ys
    
end


function geodesics(λ_final, dλ, r0, ϕ0)
    
    rs = []
    ϕs = []
    intervalo = []
    
    push!(rs, r0)
    push!(ϕs, ϕ0)
    
    u0 = -sqrt((1-2*M/r0)*(ϵ^2*(1-2*M/r0)^(-1.0)-(l/r0)^2))
    
    r = r0
    u = u0
    ϕ = ϕ0
    
    
    # Ecuación para conocer r
    fr1(λ, r, u, ϕ) = u
    # Ecuación para conocer u = dr/dλ
    fr2(λ, r, u, ϕ) = (-M*ϵ^2/r^2)*(1-2*M/r)^(-1.0) - (M*u^2)/(2*M*r-r^2) + (l^2/r^3)*(1-2*M/r)
    
    # Ecuación para conocer ϕ
    fϕ(λ, r, u, ϕ) = l/r^2
    
    λ = 0
    
    while λ < λ_final
        
        k1 = dλ*fr1(λ, r, u, ϕ) # esta es la de r
        j1 = dλ*fr2(λ, r, u, ϕ) # esta es la de u
        i1 = dλ*fϕ(λ, r, u, ϕ)  # esta es la de ϕ
        
        k2 = dλ*fr1(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1) # esta es la de u
        j2 = dλ*fr2(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1) # esta es la de r
        i2 = dλ*fϕ(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1)  # esta es la de ϕ
        
        k3 = dλ*fr1(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2) # esta es la de u
        j3 = dλ*fr2(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2) # esta es la de r
        i3 = dλ*fϕ(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2)  # esta es la de ϕ
        
        k4 = dλ*fr1(λ + dλ, r + k3, u + j3, ϕ + i3) # esta es la de u
        j4 = dλ*fr2(λ + dλ, r + k3, u + j3, ϕ + i3) # esta es la de r
        i4 = dλ*fϕ(λ + dλ, r + k3, u + j3, ϕ + i3)  # esta es la de ϕ
        
        λ = λ + dλ
        r = r + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        u = u + (1/6)*(j1 + 2*j2 + 2*j3 + j4)
        ϕ = ϕ + (1/6)*(i1 + 2*i2 + 2*i3 + i4)
        
        ds = -ϵ^2*(1-2*M/r)^(-1.0) + u^2*(1-2*M/r)^(-1.0) + (l/r)^2
        
        if r < 2*M
            break
        end
        
        push!(rs, r)
        push!(ϕs, ϕ)
        push!(intervalo, ds)
        
    end
    
    xs = []
    ys = []
    
    for i in 1:length(rs)
        push!(xs, rs[i]*cos(ϕs[i]))
        push!(ys, rs[i]*sin(ϕs[i]))
    end
    
    return xs, ys, intervalo, rs
    
end


function geodesics2(λ_final, dλ, r0, ϕ0)
    
    rs = []
    ϕs = []
    intervalo = []
    
    push!(rs, r0)
    push!(ϕs, ϕ0)
    
    u0 = -sqrt((1-2*M/r0)*(ϵ^2*(1-2*M/r0)^(-1.0)-(l/r0)^2))
    w0 = l/r0^2
    
    r = r0
    u = u0
    ϕ = ϕ0
    w = w0
    
    
    # Ecuación para conocer r
    fr1(λ, r, u, ϕ, w) = u
    # Ecuación para conocer u = dr/dλ
    fr2(λ, r, u, ϕ, w) = (-M*ϵ^2/r^2)*(1-2*M/r)^(-1.0) - (M*u^2)/(2*M*r-r^2) + r*(1-2*M/r)*w^2
    #fr2(λ, r, u, ϕ, w) = (-M*ϵ^2/r^2)*(1-2*M/r)^(-1.0) - (M*u^2)/(2*M*r-r^2) + (1-2*M/r)*(l^2/r^3)
    
    # Ecuación para conocer ϕ
    fϕ1(λ, r, u, ϕ, w) = w
    # Ecuación para conocer w = dϕ/dλ
    fϕ2(λ, r, u, ϕ, w) = -(2/r)*u*w
    
    λ = 0
    
    while λ < λ_final
        
        k1 = dλ*fr1(λ, r, u, ϕ, w) # esta es la de r
        j1 = dλ*fr2(λ, r, u, ϕ, w) # esta es la de u
        i1 = dλ*fϕ1(λ, r, u, ϕ, w) # esta es la de ϕ
        m1 = dλ*fϕ2(λ, r, u, ϕ, w) # esta es la de w
        
        k2 = dλ*fr1(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1, w + 0.5*m1) # esta es la de u
        j2 = dλ*fr2(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1, w + 0.5*m1) # esta es la de r
        i2 = dλ*fϕ1(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1, w + 0.5*m1)  # esta es la de ϕ
        m2 = dλ*fϕ2(λ + 0.5*dλ, r + 0.5*k1, u + 0.5*j1, ϕ + 0.5*i1, w + 0.5*m1)  # esta es la de w
        
        k3 = dλ*fr1(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2, w + 0.5*m2) # esta es la de u
        j3 = dλ*fr2(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2, w + 0.5*m2) # esta es la de r
        i3 = dλ*fϕ1(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2, w + 0.5*m2)  # esta es la de ϕ
        m3 = dλ*fϕ2(λ + 0.5*dλ, r + 0.5*k2, u + 0.5*j2, ϕ + 0.5*i2, w + 0.5*m2)  # esta es la de w
        
        k4 = dλ*fr1(λ + dλ, r + k3, u + j3, ϕ + i3, w + m3) # esta es la de u
        j4 = dλ*fr2(λ + dλ, r + k3, u + j3, ϕ + i3, w + m3) # esta es la de r
        i4 = dλ*fϕ1(λ + dλ, r + k3, u + j3, ϕ + i3, w + m3)  # esta es la de ϕ
        m4 = dλ*fϕ2(λ + dλ, r + k3, u + j3, ϕ + i3, w + m3)  # esta es la de w
        
        λ = λ + dλ
        r = r + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        u = u + (1/6)*(j1 + 2*j2 + 2*j3 + j4)
        ϕ = ϕ + (1/6)*(i1 + 2*i2 + 2*i3 + i4)
        w = w + (1/6)*(m1 + 2*m2 + 2*m3 + m4)
        
        if r < 2*M
            break
        end
        
        ds = -ϵ^2*(1-2*M/r)^(-1.0) + u^2*(1-2*M/r)^(-1.0) + (l/r)^2
        
        push!(rs, r)
        push!(ϕs, ϕ)
        push!(intervalo, ds)
        
    end
    
    xs = []
    ys = []
    
    for i in 1:length(rs)
        push!(xs, rs[i]*cos(ϕs[i]))
        push!(ys, rs[i]*sin(ϕs[i]))
    end
    
    return xs, ys, intervalo
    
end


function dibujar_trayectoria(xs, ys, freq=100)
    
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
    
    for θ in 0:0.05:2π + 0.05
        push!(circx, 2*M*cos(θ))
        push!(circy, 2*M*sin(θ))
    end
    
    grafica = plot(legend = false, aspect_ratio = 1)
    
    grafica = plot!(xs′, ys′)
    grafica = plot!(circx, circy, color = "black", lw = 2)
    
    return grafica
    
end  


function array_i(xs, i)
    
    xs′ = []
    
    for j in 1:i
        push!(xs′, xs[j])
    end
    
    return xs′
    
end

;
