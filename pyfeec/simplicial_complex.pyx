__all__ = ['SimplicialComplex']
from numpy import asarray, arange, empty, identity, vstack, atleast_2d, ones
from itertools import product
from scipy.sparse import csr_matrix
from scipy.linalg import solve

from .skeleton import Skeleton
from .parallel import parmap
from .grid_utils import stitch
from .barycentric cimport compute_barycentric_gradients

from cython cimport boundscheck, wraparound
from numpy cimport complex, ndarray   

class SimplicialComplex(list):
    """simplicial complex"""
        
    def __init__(self, simplices, vertices, metrics=None, stitches={}):

        simplices.sort()

        self.complex_dimension = simplices.shape[1] - 1
        self.vertices = vertices.astype("complex")
        self.embedding_dimension = vertices.shape[1]
        self.simplices = simplices
        self.points = self.vertices[self.simplices]
        self.stitches = stitches
        
        if metrics is None:
            default_metric = identity(self.embedding_dimension, dtype="complex")
            self.metrics = asarray([identity(self.embedding_dimension, dtype="complex")] * len(simplices))
        else:
            self.metrics = metrics
        
        for dim in range(self.complex_dimension):
            skeleton = Skeleton()
            skeleton.complex = self
            skeleton.dim = dim
            self.append(skeleton)
            
        skeleton = Skeleton()
        skeleton.complex = self
        skeleton.dim = self.complex_dimension
        skeleton.simplices = asarray([stitch(s, stitches) for s in simplices])
        skeleton.simplex_to_index = dict([(frozenset(stitch(simplex, stitches)), index) for index, simplex in enumerate(skeleton.simplices)])
        skeleton.num_simplices = simplices.shape[0]
        skeleton.exterior_derivative = csr_matrix((1, skeleton.num_simplices), dtype="int8")
        skeleton.boundary = asarray([], dtype="uint")
        skeleton.interior = arange(skeleton.num_simplices, dtype="uint")
        
        self.append(skeleton)

    def __repr__(self):
        output = "SimplicialComplex:\n"
        for i in reversed(range(len(self))):
            output += "   %10d: %2d-simplices\n" % (self[i].num_simplices, i)
        return output

    def __getattr__(self, attr):
        if attr == "barycentric_gradients":
            self.barycentric_gradients = asarray(parmap(lambda stuff: compute_barycentric_gradients(stuff[0], stuff[1]), zip(self.points, self.metrics)))
            return self.barycentric_gradients
        else:
            raise AttributeError(attr + " not found")