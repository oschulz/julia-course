# -*- coding: utf-8 -*-
# Check multithreading config:
Base.Threads.nthreads()

# # Instantiate package environment for this notebook
using Pkg; pkg"instantiate"

# Check active package versions:
# using Pkg; pkg"status"

# <h1 style="text-align: center;">
#     <span style="display: block; text-align: center;">
#         Introduction to
#     </span>
#     <span style="display: block; text-align: center;">
#         <img alt="Julia" src="images/logos/julia-logo.svg" style="height: 2em; display: inline-block; margin: 1em;"/>
#     </span>
#     <span style="display: block; text-align: center;">
#         Example
#     </span>
# </h1>
#
# <div style="text-align: center;">
#     <p style="text-align: center; display: inline-block; vertical-align: middle;">
#         Oliver Schulz<br>
#         <small>
#             Max Planck Institute for Physics <br/>
#             <a href="mailto:oschulz@mpp.mpg.de" target="_blank">oschulz@mpp.mpg.de</a>
#         </small>
#     </p>
#     <p style="text-align: center; display: inline-block; vertical-align: middle;">
#         <img src="images/logos/mpg-logo.svg" style="height: 5em; display: inline-block; vertical-align: middle; margin: 1em;"/>
#         <img src="images/logos/mpp-logo.svg" style="height: 5em; display: inline-block; vertical-align: middle; margin: 1em;"/>
#     </p>
# </div>
#
# <p style="text-align: center;">
#     MPI for Physics, March 2021
# </p>

# ## Example: Trajectory of a ball

# Let's analyze the slow-motion video of a bouncing ball:

videofile = "ball_throw.mp4"

# We need to load a few packages to deal with [videos](https://juliaio.github.io/VideoIO.jl), [images](https://github.com/JuliaImages/Images.jl), [colors](https://github.com/JuliaGraphics/Colors.jl) and [units](https://github.com/PainterQubits/Unitful.jl):

using VideoIO, Images, Colors, Unitful

# Let's get the duration of the video

video_duration = VideoIO.get_duration(videofile) * u"s"

# and open a video input stream:

video_input = VideoIO.openvideo(videofile)

# Get the video frame rate - `video_input.framerate` yields framerate * 1000 for some reason:

fps = VideoIO.framerate(video_input)/1000 * u"s^-1"

# Reading the whole video would use a lot of RAM. Let's write a function to read single frames at a given point in time:

typeof(video_input)

function read_frame(input::VideoIO.VideoReader, t::Number)
    t_in_s = float(ustrip(uconvert(u"s", t)))
    seek(input, t_in_s)
    read(input)
end

# Ok, now we can read a frame. We'll subsample it before display, to keep it small:

frameimg = read_frame(video_input, 3u"s")
frameimg[1:5:end,1:5:end]

# Images in in Julia (more specifically, in [Images.jl](https://github.com/JuliaImages/Images.jl)) are simply arrays of color values:

typeof(frameimg)

# But oops, the video was recorded in portrait mode, so it appears rotated. Let's transpose the image array - this will also result in a mirrored image, but that doesn't matter here:

frameimg[1:5:end,1:5:end]'

# Nice, now let's load the plotting package Plots.jl

using Plots; gr(format = :png)

# and plot a frame every second:

plot(plot.(broadcast(
    (input, t) -> read_frame(input, t)[1:5:end,1:5:end]',
    Ref(video_input),
    0u"s":1u"s":video_duration
), axis=nothing)...)

# To develop a method that detects the ball, we'll need a frame with and another frame without the ball.
#
# We'll also need image coordinates, so we'll use [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to plot the frames with a coordinate system. Let's load Plots and select the [GR.jl](https://github.com/jheinen/GR.jl) backend, which create plots via the [GR Framework](https://gr-framework.org/):

# We won't flip/transpose the images this time, so that we don't confuse the images axes later on:

background_frame, ball_frame = read_frame.(Ref(video_input), [0, 3]u"s")

plot(
    plot(background_frame, xlabel = "j", ylabel = "i"),
    plot(ball_frame, xlabel = "j", ylabel = "i"),
)

# Note that Plots.jl plots images with matrix-like row/column direction.
#
# Each frame image/array is a 1080 x 1920 matrix, with indices 1:1080 for the rows and 1:1920 for the columns:

size(background_frame)

axes(background_frame)

# To find the ball in the video, we need it's color. Let's zoom into `ball_frame`:

plot(ball_frame[180:260,90:170], ratio = 1)

# And zoom in some more, until only ball color is left:

plot(ball_frame[202:238,112:148], ratio = 1)

# We want the average color in that image region. To calculate means, we need the [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics) package, which is part of the Julia standard library:

using Statistics

# Since images are arrays, we can simply use the function `mean` to get the average color. We convert the color to the HSV color space, since it should be easiest to locate the ball based on color:

ball_color = HSV{Float32}(mean(ball_frame[205:235,115:145]))

# We define the distance between two colors based on the difference in hue and saturation:

function color_dist(a::Color, b::Color)
    @fastmath begin
        ca = convert(HSV, a)
        cb = convert(HSV, b)
        sqrt((Float32(ca.h - cb.h)/360)^2 + Float32(ca.s - cb.s)^2)
    end
end

# Using `color_dist`, we can define the difference between two frames:

framediff(f::Function, frame::AbstractArray, ref_frame::AbstractArray, ref_color::Color) =
    f.(color_dist.(frame, ball_color) .- color_dist.(ref_frame, ball_color))

framediff(frame::AbstractArray, ref_frame::AbstractArray, ref_color::Color) =
    framediff(identity, frame, ref_frame, ref_color)

# Let's see how this performs:

typeof(similar(background_frame))

heatmap(framediff(ball_frame, background_frame, ball_color))

# Not bad - looks like a threshold of -0.4 might be a good choice to separate pixels belonging to the ball from pixels belonging to the background:

heatmap(framediff(x -> x < -0.4, ball_frame, background_frame, ball_color))

# That looks like a clean cut. Now all we need to do is to process the whole video. We generate the pixel masks on the fly, to avoid storing the whole video in RAM. Let's define a function for this, in case it needs to be re-run this a different reference color or threshold. Also, let's use multi-threading to process video frames in parallel:

using Base.Threads

function process_video(input::VideoIO.VideoReader, bg_frame::AbstractMatrix, fg_color::Color, threshold::Real)
    seek(input, 0.0)

    first_frame = read(input)
    result = [framediff(x -> x < threshold, first_frame, bg_frame, fg_color)]
    
    result_lock = ReentrantLock()

    input_channel = Channel{Tuple{Int, typeof(first_frame)}}(nthreads(), spawn = true) do ch
        i = length(result)
        while !eof(input)
            i += 1
            push!(ch, (i, read(input)))
        end
    end    
    
    @sync for _ in 1:nthreads()
        @Base.Threads.spawn for (i, frame) in input_channel
            r = framediff(x -> x < threshold, frame, bg_frame, fg_color)
            lock(result_lock) do
                nframes = max(length(result), i)
                resize!(result, nframes)
                result[i] = r
            end
        end
    end
    
    @assert all(isassigned.(Ref(result), eachindex(result)))
   
    result
end

diffvideo = @time process_video(video_input, background_frame, ball_color, -0.4)
typeof(diffvideo), length(diffvideo)

heatmap(diffvideo[50])

# We interpret each difference frame as a matrix of weights (0 or 1) and estimate the position of the ball as the weighted mean of image coordinates/indices. [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) will come in handy here to handle vectors of fixed size that can be stack-allocated:

using StaticArrays

function mean_pos(W::AbstractArray{T,N}) where {T,N}
    U = float(T)
    R = SVector{N,U}
    sum_pos::R = zero(R)
    sum_w::U = zero(U)
    @inbounds for idx in CartesianIndices(W)
        w = W[idx]
        sum_pos += SVector(Tuple(idx)) * w
        sum_w += w
    end
    sum_pos / sum_w
end

# Let's see if the is fast enough, using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl):

using BenchmarkTools

@benchmark mean_pos($diffvideo[1])

# That should do, speed-wise!
#
# StaticArrays.jl allows us to define custom field-vector types, we'll need something to represent 2D x/y vectors:

struct Vec2D{T} <: FieldVector{2,T}
    x::T
    y::T
end

# Now we can reconstruct the ball positions, as a vector of `Vec2D`:

Vec2D.(mean_pos.(diffvideo))

# However, we'll also frequently want to access all `x` and `y` fields as separate vectors.  [StructArrays.jl](https://github.com/JuliaArrays/StructArrays.jl) allows us to store this data as a [Structure of Arrays](https://en.wikipedia.org/wiki/AoS_and_SoA), with both AoS and SoA semantics:

using StructArrays

PV = StructArray(Vec2D.(mean_pos.(diffvideo)))
typeof(PV.x), typeof(PV.y), typeof(PV[1])

# Did we reconstruct the ball positions correctly?

plot(PV.x, PV.y, yflip = true)

# That looks promising. We also need a time axis, though - let's use a [`TypedTables.Table`](https://github.com/JuliaData/TypedTables.jl) to put it all together:

using TypedTables

realtime_framerate = 240
raw_data = Table(xy = PV, t = (eachindex(PV) .- firstindex(PV)) / realtime_framerate)

# Note: A [`DataFrames.DataFrame`](https://github.com/JuliaData/DataFrames.jl) would also do, we choose `TypedTables.Table` here for type stability.
#
# Let's pull in [Interact.jl](https://github.com/JuliaGizmos/Interact.jl) for interactive data exploration. We'll also use [Printf](https://docs.julialang.org/en/v1/stdlib/Printf/) from the Julia standard library for number formatting.


# In the following, we'll only analyse the fist arc of the trajectory:

sel_idxs = 27:244

# FileIO.save("background.png", background_frame)

raw_xy_shift = Vec2D(0, lastindex(axes(background_frame,2)))

xy_cal_factor = 1.83 / 1559 * 1.72/1.82

xy_cal = SMatrix{2,2}(
    xy_cal_factor,             0,
                0, -xy_cal_factor
)

cal_data = Table(
    xy = StructArray(Vec2D.(Ref(xy_cal) .* (raw_data.xy .- Ref(raw_xy_shift)))),
    t = copy(collect(raw_data.t)),
)

# Fix missing frame:
view(cal_data.t, 170:lastindex(cal_data.t)) .+= 1 / realtime_framerate

sel_data = cal_data[sel_idxs]

scatter(
    sel_data.xy.x, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0)),
    xlabel = "x [m]", ylabel = "y [m]"
)

using CurveFit

f = curve_fit(CurveFit.Polynomial, sel_data.t, sel_data.xy.y, 2)

scatter(
    sel_data.xy.x, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0)),
    xlabel = "x [m]", ylabel = "y [m]"
)

plot(sel_data.t, f.(sel_data.t), label = "fit")
scatter!(
    sel_data.t, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0)),
    xlabel = "t [s]", ylabel = "y [m]",
    label = "data"
)

g_curvefit = -2 * f.coeffs[3] * u"m/s"

# ### Bayesian inference of motion parameters

# In the following, we'll need [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) from the Julia standard library, [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) from the [Julia differential equations suite](https://docs.juliadiffeq.org/) and [ValueShapes.jl](https://github.com/oschulz/ValueShapes.jl):

using LinearAlgebra, StaticArrays, OrdinaryDiffEq
using Statistics, StatsBase, Distributions, ValueShapes, InverseFunctions, BAT

function motion_eqn!(du::AbstractVector, u::AbstractVector, p::AbstractVector, t::Real)
    x, y, dx, dy = u
    ρ, A, m, g, C_w = p

    xy = SVector(x, y); d_xy = SVector(dx, dy)
    
    f_drag = - ρ * A * C_w * norm(d_xy) * d_xy / 2
    f_grav = m * SVector(zero(g), -g)
    dd_xy = (f_drag + f_grav) / m

    du .= (d_xy[1], d_xy[2], dd_xy[1], dd_xy[2])
    return du
end

function simulate_motion(v::NamedTuple, timesteps::AbstractVector = 0:0.05:1)
    u0 = [v.x, v.y, v.vx, v.vy]
    p = [v.ρ, v.A, v.m, v.g, v.C_w]

    odeprob = ODEProblem{true}(motion_eqn!, u0, (first(timesteps), last(timesteps)), p)

    sol = solve(odeprob, Tsit5(), saveat = timesteps)
    (x = sol[1,:], y = sol[2,:], t = timesteps)
end


likelihood = let data = (x = sel_data.xy.x, y = sel_data.xy.y, t = sel_data.t)
    v -> begin
        σ_x, σ_y  = v.noise .^ 2
        sim_data = simulate_motion(v, data.t)
        (log = sum(logpdf.(Normal.(sim_data.x, σ_x), data.x)) + sum(logpdf.(Normal.(sim_data.y, σ_y), data.y)),)
    end
end


prior = NamedTupleDist(
    x = Normal(0, 1),
    y = Normal(1, 2),
    vx = Normal(1, 1),
    vy = Normal(2, 2),
    ρ = 1.209, # air density at 22°C and 1024 mbar, in kg/m^3
    A = pi * (60e-3/2)^2, # ball cross section area
    m = 7.1e-3, # mass of ball, in kg
    g = Weibull(250, 9.8), # 9.81
    C_w = Weibull(20, 0.5), # unitless, 0.47 would be a typical value for a sphere
    noise = [sqrt(0.01), sqrt(0.01)]
    #noise = [Weibull(1, 0.005), Weibull(1, 0.005)] # noise (stderr)
)

posterior = PosteriorDensity(likelihood, prior)

logvalof(posterior)(rand(prior))

@benchmark logvalof(posterior)(rand(prior))

plt = scatter(
    sel_data.xy.x, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0))
)
for xy in simulate_motion.(rand(prior, 100))
    plot!(xy.x, xy.y, color = :lightblue, legend = false)
end
plt

v_guess = rand(prior)

using ForwardDiff
let vs = varshape(prior)
    ForwardDiff.gradient(v -> likelihood(vs(v)).log, inverse(vs)(v_guess))
end

# Simple maximum likelihood:

using Optim
let vs = varshape(prior)
    r = Optim.optimize(v -> - likelihood(vs(v)).log, inverse(vs)(v_guess), Optim.LBFGS(); autodiff = :forward)
    varshape(prior)(Optim.minimizer(r))
end

# Maximum posterior estimate:

findmode_ret = bat_findmode(posterior, MaxDensityLBFGS())
findmode_ret.info

mode_est = findmode_ret.result

sim_data_bestfit = simulate_motion(mode_est, sel_data.t)
plot(
    sim_data_bestfit.x, sim_data_bestfit.y,
    #=marker = (:circle, 2, stroke(0)),=#
    label = "best fit"
)
scatter!(
    sel_data.xy.x, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0)),
    xlabel = "x [m]", ylabel = "y [m]",
    label = "data"
)

ENV["JULIA_DEBUG"] = "BAT"

samling_output = @time bat_sample(posterior, MCMCSampling(mcalg = HamiltonianMC(), nchains = 4, nsteps = 10^4))
samples = samling_output.result;

plot(samples)

plot(samples, vsel = [:vx, :vy, :C_w, :g])

SampledDensity(posterior, samples)

mode_samples = bat_findmode(samples).result

mode_refined = bat_findmode(posterior, MaxDensityLBFGS(init = ExplicitInit([mode_samples]))).result

sim_data_bestfit = simulate_motion(mode_refined, sel_data.t)
plot(
    sim_data_bestfit.x, sim_data_bestfit.y,
    #=marker = (:circle, 2, stroke(0)),=#
    label = "best fit, refined"
)
scatter!(
    sel_data.xy.x, sel_data.xy.y,
    marker = (:circle, 2, :black, stroke(0)),
    xlabel = "x [m]", ylabel = "y [m]",
    label = "data"
)

using DensityInterface, InverseFunctions, ChangesOfVariables
using Test

tpstr, trafo = bat_transform(PriorToGaussian(), posterior)
v = rand(prior)
x = trafo(v)

logdensityof(tpstr)(x)
ForwardDiff.gradient(logdensityof(tpstr), x)

@inferred logdensityof(tpstr)(x)
@inferred ForwardDiff.gradient(logdensityof(tpstr), x)

@benchmark logdensityof($tpstr)($x)
@benchmark ForwardDiff.gradient(logdensityof($tpstr), $x)
