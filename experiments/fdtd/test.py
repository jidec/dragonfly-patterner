import fdtd

fdtd.set_backend("torch")

grid = fdtd.Grid(
    shape = (25e-6, 15e-6, 1), # 25um x 15um x 1 (grid_spacing) --> 2D FDTD
)

#grid[11:32, 30:84, 0] = fdtd.Object(permittivity=1.7**2, name="object")

grid[13e-6:18e-6, 1e-6:14e-6, 0] = fdtd.Object(permittivity=1.5**2)

grid[7.5e-6:8.0e-6, 11.8e-6:13.0e-6, 0] = fdtd.LineSource(
    period = 1550e-9 / (3e8), name="source"
)

grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")
#print(grid.detector)

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

print(grid)

grid.run(total_time=100)

# signature
grid.visualize(z=0,show=True)

