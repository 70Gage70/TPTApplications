# TPT Applications

To run the included notebook, follow these instructions.

### 1. 

Install [Julia](https://julialang.org/). On windows, execute

```
winget install julia -s msstore
```

in a command line. On Mac and Linux, run

```
curl -fsSL https://install.julialang.org | sh
```

in a shell. Not working? Follow [these instructions](https://github.com/JuliaLang/juliaup).

### 2. 

Open julia by typing `julia` in a terminal. Copy and paste the following code to the prompt:

```julia
using Pkg
Pkg.add("Pluto")
using Pluto
begin
path = "https://raw.githubusercontent.com/70Gage70/TPTApplications/main/notebook.jl"
nb = download(path)
Pluto.run(notebook = nb)
end
nothing
```

After a few minutes of calculations, this will open a browser window containing the notebook. It may take a minute or two to fully load.

