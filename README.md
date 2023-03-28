# Introduction to Julia

This course currently constists of

* [`"julia-course-slides.ipynb"`](julia-course-slides.ipynb): Slides as as Jupyter notebook

* [`"julia-course-example.ipynb"`](julia-course-example.ipynb): A practical example as a Jupyter notebook

You need both Julia and a way to run Jupyter notebooks to run this course.


## Installing Julia and either Jupyter or nteract

### Installing Julia

Julia is easy to install:

* [Download Julia](https://julialang.org/downloads/).

* Extract the archive resp. run the installer.

* You may want to add the Julia "bin" directory to your `$PATH"`

We highly recommend using Julia v1.9 to run the code in this course.


### Installing Jupyter

If you have a working Jupyter installation, it should detect the Jupyter Julia kernel (see below on how to install it) automatically.

You can also start Jupyter via Julia: This can either use existing installations of Jupyter, or install both internally by creating an internal Conda installation within `$HOME/.julia/conda`. On Linux, Julia will by default to use the Jupyter installation associated with the `jupyter` executable on your `$PATH`. On OS-X and Windows, both IJulia will by default always create a Julia-internal Conda installation (see above). To change this behavior, set the environment variable [`$JUPYTER`](https://github.com/JuliaLang/IJulia.jl#installation). For details, see the [IJulia.jl](https://github.com/JuliaLang/IJulia.jl#installation)documentation.

Note: This course doesn't call on any Python packages from Julia. If you *do* call Python from Julia though (e.g. indirectly via packages like PyPlot.jl and UltraNest.jl or directly via PyCall.jl), the same default behavior occurs (the system's Python3 is used on Linux, a Julia-internal Conda environment on OS-X and Windows). To change this, set the [`$PYTHON`](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) environment variable. For details, see the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) documentation.

If you want to use a standalone Jupyter/Python installation with Julia, we recommend [installing Anaconda](https://www.anaconda.com/distribution/).


### Jupyter alternative: Installing nteract

On local systems, you can use the [nteract](https://nteract.io/) deskop application to run Jupyter notebooks, instead of using a Jupyter server. Like Jupyter, nteract should detect the Jupyter Julia kernel (see below) automatically.


### Environment variables

You may want/need to set the following environment variables:

* `$PATH`: Include the Julia `bin`-directory in your binary search path, see above.
If you intend to use Jupyter, you will probably want to include the directory containing the `jupyter` binary to your `PATH` as well.

* [`$JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS-1): Number of threads to use for Julia multi-threading

* [`$JULIA_DEPOT_PATH`](https://julialang.github.io/Pkg.jl/v1/glossary/) and [`JULIA_PKG_DEVDIR`](https://julialang.github.io/Pkg.jl/v1/managing-packages/#Developing-packages-1): If you want Julia to install packages in another location than `$HOME/.julia`.

See the Julia manual for a description of [other Julia-specific environment variables](https://docs.julialang.org/en/v1/manual/environment-variables/).


### Installing the Jupyter Julia kernel

First install the [IJulia Jupyter Julia kernel](https://github.com/JuliaLang/IJulia.jl), [Interact.JL](https://github.com/JuliaGizmos/Interact.jl) and [WebIO.jl](https://github.com/JuliaGizmos/WebIO.jl) in your *default* Julia project environment via

```shell
julia -e 'using Pkg; Pkg.add(["IJulia", "Interact", "WebIO"]); Pkg.build("IJulia")'
```

Also run

```shell
julia -e 'using WebIO; WebIO.install_jupyter_nbextension()'
```

to install a Jupyter extension that [WebIO.jl](https://github.com/JuliaGizmos/WebIO.jl) (used by Interact.jl) requires to function.

To configure Julia to use multiple threads when run as a Jupyter kernel, use

```shell
julia -e 'using IJulia; IJulia.installkernel("Julia", "--project=@.", "--threads=auto")'
```

On Julia versions *older than v1.6*, you need to use

```shell
julia -e 'using IJulia; IJulia.installkernel("Julia", "--project=@.", env=Dict("JULIA_NUM_THREADS"=>"4"))'
```

instead.


## Setting up this course

Download this course via `Git` and change into the "julia-course" directory:

```shell
git clone https://github.com/oschulz/julia-course.git
cd julia-course
```

Julia has a very powerful [package management system](https://julialang.github.io/Pkg.jl/v1/) that allows for using different versions of packages for different projects, layered package environments, etc. Run the shell command

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

to instantiate the [Julia project environment](https://docs.julialang.org/en/v1/manual/code-loading/#Project-environments-1) defined by the files "Project.toml" and "Manifest.toml" in the "julia-course" directory.

Note that the "IJulia" package should always be installed in the *default environment* (see above) and *not* in individual project environments, to avoid version conflicts (since the Jupyter kernel will always try to load the same one).

Optional: To make this environment provided with this course your default Julia environment, typically located in `"$HOME/user/.julia/environments/v1.6"`,
simply copy the files `"Project.toml"` and `"Manifest.toml"` there, and then add IJulia (see above).


## Using the Jupyter notebooks

First ensure that you have the "IJulia" package installed, which provides the Jupyter Julia kernel. Test by running (should not report an error)

```shell
julia -e 'using IJulia'
```

If you do *not* have a Jupyter installation on your `$PATH`, you may want to start [Jupyter via Julia](https://julialang.github.io/IJulia.jl/stable/manual/running/) or (on a desktop system) use [nteract](https://nteract.io/).

If you *do* have a Jupyter installation on your `$PATH` (preferred), you can usually just start a [Jupyter notebook server](https://jupyter-notebook.readthedocs.io/en/stable/) using

```shell
jupyter notebook
```

When using a Jupyter installation on your local system, your web browser will usually be started automatically and be pointed to the Jupyter notebook server instance. However, when using a software container or when starting Jupyter on a remote system using SSH port forwarding (and in some other cases), Jupyter will complain that it can't start a web browser. In these cases, run

```shell
jupyter notebook --no-browser
```

Jupyter will print the URL to point your web browser too. That URL should include an authorization token (unless you configured Jupyter for [password-based access](https://jupyter-notebook.readthedocs.io/en/stable/security.html#alternatives-to-token-authentication)).

Depending on where and how you run Jupyter - especially if you run in a Docker container - you may need to specify a non-standard port number and/or IP address to bind to, or allow Jupyter to run in a root user account. In such cases, additional options will be required, e.g.:

```shell
jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root
```
