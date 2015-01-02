pyfeec is a discrete exterior calculus framework for Hermitian manifolds.

Please refer to http://arxiv.org/abs/1405.7879 for a full description.

#Meshes

pyfeec provides three ways of triangulating an N dimensional cube:
* Symmetric Grid
* Asymmetric Grid
* Random Mesh

##Symmetric and Asymmetric Grids

The symmetric and asymmetric grid results in a mesh documented at http://arxiv.org/abs/1405.7879.

To create a(n) (a)symmetric, you will first need to create a dictionary of grid indices.
This is a dictionary that maps a set of coordinate indices (grid indices) to the index of the corresponding vertex (vertex indices).
For example, the origin's grid index would be (0,0,...,0), and might be mapped to the first vertex, vertex 0.
One step in the Nth direction would yeild a grid index of (0,0,...,1).
This might be mapped to vertex 1, and so on.

Use pyfeec.grid_indices([M1, M2,..., MN]) to create a dictionary of grid indices for an M1xM2x...xMN grid.

**Example**
```python
import pyfeec
shape = [4,5,3]
grid_indices = pyfeec.grid_indices(shape)
```

The space can now be divided into simplices using the symmetric or asymmetric grid functions (pyfeec.symmetric_grid and pyfeec.grid respectively).
In the above example, a 4x5x3 grid is used.

**Example**
```python
symmetric_grid_simplices = pyfeec.grid(grid_indices)
asymmetric_grid_simplices = pyfeec.asymmetric_grid(grid_indices)
```

Both functions return a 2D numpy array of simplices. Each row is a list of vertex indices that forms a simplex.


##Random Meshes

Random meshes are created without grid indices.
Instead, pyfeec's random\_mesh function takes the number of points on the interior of the mesh and the dimension of the mesh as arguements.

**Example**
```python
vertices, stitches = pyfeec.random_mesh(150, 3)
```

In the above example, the mesh will be 3 dimensional and have have 150 points on its interior. vertices is a 2D numpy array of coordinates.
The ith row gives the coordinates of the ith vertex. In the above usage, stitches will be an empty python dictionary.
random_mesh can take a third argument for imposing periodic boundary conditions.
This will result in stitches becomming a nonempty dictionary and will be discussed in the next section.

random\_mesh can also take a keyword (or fourth argument), alpha. This parameter determines how close the points can be to each other. Statistically speaking, the points would be an average distance of N ^ (1 / dim) apart, wtih N being the number of points on the interior and dim being the dimension of the complex. random_mesh will ensure the points are no closer than alpha * N ^ (1 / dim) apart. alpha is set to 0.5 by default. Increasing alpha will increase the time it takes to randomly generate a suitable set of points.

#Customizing Topology

Given a triangulated N dimensional cube, it possible to create any desired topology by gluing or "stitching" vertices together.
When using pyfeec, these "stitches" are stored as a dictionary that maps vertex indices to the corresponding vertex indices they should be stitched to.
To create a torus for periodic boundary conditions, points on opposite sides of the grid are stitched together.
pyfeec comes with a function called pbc_stitches that creates the stitches necessary to impose periodic boundary conditions in any combination of directions.

**Example**
```python
stitches = pyfeec.pbc_stitches(grid_indices, shape, [0, 2])
```

The third argument is a list of (zero indexed) directions that should be periodic.
In the above example, the first and third directions are to be periodic. For random meshes, this list of directions is specified as a third argument.

**Example**
```python
vertices, stitches = pyfeec.random_mesh(150, 3, [0, 2])
```

#Embedding the Vertices

The next step is choosing a coordinate system to embed the vertices.

##Symmetric and Asymmetric Grids
The embed function is used to create a list of coordinates from grid indices.

**Example**
```python
from numpy import linspace
coordinates = [linspace(0,1,s) for s in shape]
vertices = pyfeec.embed(grid_indices, coordinates)
```

Here, coordinates are the coordinates of each N-1 dimensional hyperplane that discretizes your N dimensional cubic space. For example, say

```python
shape=[2, 3, 2]
```

A uniform mesh on the unit cube is generated with the following hyperplane coordinates:

```python
[
 [0, 1],
 [0, 0.5, 1],
 [0, 1]
]
```

This means that the x-axis is discretized with following coordinates:
x=0, x=1
The y-axis is discretized with the following coordinates:
y=0, y=0.5, y=1
The z-axis is discretized with the following coordinates:
z=0, z=1

Say there is more interesting phenomina near y=1. Then one might choose to use a nonuniform mesh with the following coordinates:

```python
[
 [0, 1],
 [0, 0.8, 1],
 [0, 1]
]
```

##Random Meshes
Random meshes automatically embed vertices on the unit N-cube. The coordinates of these vertices are returned as shown below:

```python
vertices, stitches = pyfeec.random_mesh(150, 3, [0,2])
```

#Creating a Simplicial Complex

Simplicial complexes are created with the SimplicialComplex class. It can be initialized as follows:

```python
sc = pyfeec.SimplicialComplex(simplices, vertices)
```

For a customized topology, use

```python
sc = pyfeec.SimplicialComplex(simplices, vertices, stitches=stitches)
```

#Specifying a Metric

By default, the Euclidian metric is used.
The metric takes the form of a DxD hermitian matrix (for a D dimensional embedding) at each N-dimensional simplex (where N is the complex dimension). 
This can be specified in the form of one of the following:

* a MxDxD array: for a complex composed of M N-dimensional simplices.
* a DxD array: for a homogeneous complex. This metric is assumed to be constant throughout the complex.
* a D array: for a homogenous complex with a diagonal metric.

#de Rham maps
The de Rham map returns a two element list containing the average of the input function over the domain of each simplex in the complex.
See http://arxiv.org/abs/1405.7879 for more on this process.
In general, the pyfeec's de Rham map uses a generalized trapeziodal rule.
The number of points sampled along each direction during this process is specified with the keyword, subdivisions.

**Example** (the metric for polar coordinates)

```python
from numpy import asarray
def g(pt):
    r, theta = pt
    m = asarray([
                 [1, 0],
                 [0, r ** 2]
                ], dtype="complex")
    return m
    
metrics = pyfeec.derham_map(vertices[simplices], g, subdivision=3)
sc = pyfeec.SimplicialComplex(simplices, vertices, stitches=stitches, metrics=metrics)
```

#Cohomological Operators

Simplicial complexes are really a list of p-skeletons. Each p-skeleton computes and stores information related to the p-skeleton (set of p-simplicies) of the simplicial complex.
These properties are computed as needed--an idea borrowed from PyDEC.
The following properties are generally of the most interest:

* exterior_derivative
* star
* inverse_star
* codifferential = star dot exterior_derivative
* laplace_beltrami = codifferential dot exterior_derivative
* laplace_derham = laplace_beltrami + exterior_derivative dot codifferential

All operators are stored as a sparse csr_matrix (see scipy docs).

For example, the laplacian for scalar fields is given by:

```python
L_scalar = sc[0].laplace_beltrami
```

The laplacian for vector fields is then given by:

```python
L_vector = sc[1].laplace_beltrami
```

And so on.

#Sharpening a Differential Form

Each p-skeleton has a method called sharpen that can be used to turn a differential p-form into a antisymmetric p-rank tensor field.
Given a p-form (stored as a one dimensional numpy array of coefficients of p-simplices), one can use sharpen as follows:

```python
tensor_field = sc[p].sharpen(p_form)
```

Each tensor in the tensor field is located at the barycenter of a corresponding N-simplex.

#Visualizing Fields

pyfeec comes with two tools to visualize 2D fields. Tools for animating 2D fields to visualize 3D ones may come soon.
To visualize scalar fields, use the scalar_field2d function. This returns a matplotlib image of the field interpolated.

**Example**
```python
from matplotlib.pylab import show 

barycenters = sc.points.mean(1)
scalar_field = sc[0].sharpen(form)

pyfeec.scalar_field2d(barycenters, scalar_field, 1000)
show()
```

This will interpolate the scalar field over a 1000x1000 grid.

Similarly, one can use vector_field2d to visualize a vector field.

**Example**
```python
from matplotlib.pylab import show 

barycenters = sc.points.mean(1)
vector_field = sc[1].sharpen(form)

pyfeec.vector_field2d(barycenters, vector_field, 20)
show()
```
