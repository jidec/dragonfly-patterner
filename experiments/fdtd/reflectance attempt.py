import fdtd

# use fdtd to, have visualized structural elements in a surface in the lab, use these assumptions to
# model a reflectance spectrum and compare it to the real reflectance spectrum
# if they are very similar, it is likely our assumptions are correct

# could also model a bunch of known arrangements and try to find one that produced a close fit to the spectrum?
# for example, given a reflectance spectrum (estimated from a camera response OR obtained in the lab)
#   1. get pigment presences by spectral signatures
#   2. model multiple known arrangements using present pigments
#   3. pick the best fitting arrangement and say that is the most likely sturctural color in the color cluster
# assess dispersion of color clusters to get angle-dependent color

fdtd.set_backend("torch")

# 2d grid
grid = fdtd.Grid(
    shape = (25e-6, 15e-6, 1), # 25um x 15um x 1 (grid_spacing) --> 2D FDTD
)

# reflecting object
grid[13e-6:18e-6, 1e-6:14e-6, 0] = fdtd.Object(permittivity=1.5**2)

# light source
grid[15, :] = fdtd.LineSource(
   period = 1550e-9 / (3e8), name="source"
)
#grid[15, :] = fdtd.LineSource(period=WAVELENGTH / SPEED_LIGHT, name="source")

grid[15, :] = fdtd.LineDetector(name="detector")
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