import pyfeec
from numpy import linspace, asarray, identity, zeros, pi
from numpy.random import rand
from scipy.linalg import eigh
from matplotlib.pylab import show

N = 20
dim = 2

shape = [N]*dim

grid_indices = pyfeec.grid_indices(shape)
simplices = pyfeec.grid(grid_indices)
coordinates = [linspace(0,pi,s) for s in shape]
vertices = pyfeec.embed(grid_indices, coordinates)
sc = pyfeec.SimplicialComplex(simplices, vertices)

interior = sc[0].interior
#interior = range(sc[0].num_simplices) #uncomment for Nuemann conditions
K = sc[0].laplace_beltrami[interior, :][:, interior]

val0, vec0 = eigh(K.toarray(), sc[0].star.toarray())
vec0 = vec0.T

values = [(val.real, vec) for val, vec in zip(val0, vec0) if val.real > .9 and abs(val.imag) < 0.1]
values.sort(key=lambda x: x[0])
eigenvalues = []
eigenvectors = []
for i,value in enumerate(values):
    eigenvalues.append(value[0])
    eigenvectors.append(value[1])
eigenvalues = asarray(eigenvalues)
eigenvectors = asarray(eigenvectors)


n = 2

barycenters = sc.points.mean(1).real
z = zeros(sc[0].num_simplices)
z[interior] = eigenvectors[n].real

pyfeec.scalar_field2d(barycenters, sc[0].sharpen(z).real, 1000)

show()