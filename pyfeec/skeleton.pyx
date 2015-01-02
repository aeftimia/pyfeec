__all__ = ['Skeleton']

from numpy import asarray, zeros, ones, empty, sqrt, arange, int8, diag
from scipy.sparse import csr_matrix, dia_matrix, lil_matrix, coo_matrix
from scipy.sparse.linalg import inv
from scipy.misc import factorial
from itertools import combinations
from functools import reduce
from operator import or_

from .form_utils import wedge, naive_derham_map
from .grid_utils import stitch
from .parallel import parmapreduce, parmap
from .circumcenter cimport compute_circumcenter
from .linalg cimport det

from cython cimport boundscheck, wraparound
from numpy cimport complex, ndarray

class Skeleton(_Skeleton):
    """caches the result of costly operations"""
    
    def __getattr__(self, attr):
        if attr == "sharp":
            self.compute_sharp()
            return self.sharp
        elif attr == "boundary":
            if self.dim == self.complex.complex_dimension - 1:
                self.boundary = asarray(((abs(self.exterior_derivative).sum(axis=0).getA1() % 2) == 1).nonzero()[0], dtype="uint")
            else:
                old_boundary = self.complex[self.dim + 1].boundary
                if len(old_boundary):
                    self.boundary = asarray(abs(self.exterior_derivative[old_boundary]).sum(axis=0).getA1().nonzero()[0], dtype="uint")
                else:
                    self.boundary = asarray([], dtype="uint")
            return self.boundary
        elif attr == "interior":
            self.interior = asarray(sorted(set(arange(self.num_simplices, dtype="uint")) - set(self.boundary)), dtype="uint")
            return self.interior
        elif attr in ['exterior_derivative', 'simplices', 'simplex_to_index', 'num_simplices']:
            self.compute_exterior()
            return getattr(self, attr)
        elif attr == "star":
            self.star = self.sharp.conj().T.dot(self.sharp) / factorial(self.dim)
            return self.star
        elif attr == "inverse_star":
            self.inverse_star = inv(self.star)
            return self.inverse_star
        elif attr == "codifferential":
            if self.dim:
                self.codifferential = (-1) ** self.dim * self.complex[self.dim - 1].inverse_star.dot(self.complex[self.dim - 1].exterior_derivative.T).dot(self.star)
            else:
                self.codifferential = zeros(self.num_simplices)
            return self.codifferential
        elif attr == "laplace_beltrami":
            if self.dim < self.complex.complex_dimension:
                self.laplace_beltrami = self.complex[self.dim + 1].codifferential.dot(self.exterior_derivative)
            else:
                self.laplace_beltrami = zeros((self.num_simplices,) * 2)
            return self.laplace_beltrami
        elif attr == "laplace_derham":
            self.laplace_derham = self.laplace_beltrami.copy()
            if self.dim:
                self.laplace_derham = self.laplace_derham + self.complex[self.dim - 1].exterior_derivative.toarray().dot(self.codifferential)
            return self.laplace_derham
        else:
            raise AttributeError(attr + " not found")

cdef class _Skeleton(object):
    @boundscheck(False)
    @wraparound(False)
    cpdef compute_exterior(self):
        cdef ndarray[long unsigned int, ndim=2] old_simplices = asarray(list(self.complex[self.dim + 1].simplices))
        cdef long unsigned int col, num_simplices = 0, col_max = old_simplices.shape[0]
        cdef unsigned char j, j_max = old_simplices.shape[1]
        cdef list simplex, face
        cdef frozenset sface
        cdef list indices = []
        cdef list indptr = [0]
        cdef list data = []
        cdef list simplices = [], points = []
        cdef dict simplex_to_index = {}
        for col in range(col_max):
            simplex = list(old_simplices[col])
            for j in range(j_max):
                face = simplex[:j] + simplex[j + 1:]
                sface = frozenset(face)
                if sface in simplex_to_index:
                    indices.append(simplex_to_index[sface])
                else:
                    simplex_to_index[sface] = num_simplices
                    indices.append(num_simplices)
                    simplices.append(face)
                    num_simplices += 1
                data.append(int8((-1) ** j))
            indptr.append((col + 1) * j_max)
        self.simplices = asarray(simplices, dtype="uint")
        self.exterior_derivative = csr_matrix((data, indices, indptr), (old_simplices.shape[0], self.simplices.shape[0]), dtype="int8")
        self.num_simplices = num_simplices
        self.simplex_to_index = simplex_to_index

    @boundscheck(False)
    @wraparound(False)
    def compute_sharp(self):
        cdim = self.complex.complex_dimension + 1
        edim = self.complex.embedding_dimension
        dim = self.dim + 1
        step =  edim ** self.dim
        combos = asarray(list(combinations(range(cdim), dim)), dtype="uint8")
        subset_indices = [(asarray([j for j in range(dim) if j != i0], dtype="uint8"), (-1) ** i0) for i0 in range(dim)]
        normalize = len(combos) * len(subset_indices) * factorial(cdim - dim)
        simplices = self.complex[cdim - 1].simplices
        all_barycentric_gradients = self.complex.barycentric_gradients

        if not self.dim:
            rows = []
            columns = []
            data = []
            for n in range(self.complex[cdim - 1].num_simplices):
                for j in range(cdim):
                    rows.append(n)
                    columns.append(self.simplex_to_index[frozenset(stitch((simplices[n, j],), self.complex.stitches))])
                    data.append(1)
            self.sharp = coo_matrix((data, (rows, columns)), (self.complex[cdim - 1].num_simplices, self.num_simplices)).tocsr() / normalize
            return
        
        def compute_sharp(n):
            sharp = lil_matrix((self.complex[cdim - 1].num_simplices * step, self.num_simplices), dtype="complex")
            simplex = simplices[n]
            barycentric_gradients = all_barycentric_gradients[n]
            n *= step
            for combo in combos:
                vectors = barycentric_gradients[combo]
                i = self.simplex_to_index[frozenset(stitch(simplex[combo], self.complex.stitches))]
                for subset_index, c in subset_indices:
                    for j, element in enumerate(reduce(wedge, vectors[subset_index]).flat):
                        if element != 0:
                            sharp[n + j, i] += element * c
            return sharp

        self.sharp = parmapreduce(compute_sharp, range(self.complex[cdim - 1].num_simplices)).tocsr() / normalize
        
    cpdef sharpen(self, form):
        return self.sharp.dot(form).reshape((self.complex[-1].num_simplices,) + (self.complex.embedding_dimension,) * self.dim)
        
    def _compute_star(self, stuff):
        star = zeros((self.num_simplices,) * 2, dtype='complex')
        differentials, metric, n_simplex = stuff
        p_simplices = asarray(list(combinations(range(self.complex.complex_dimension + 1), self.dim + 1)), dtype='uint8')
        bg = differentials[p_simplices]
        p_indices = [self.simplex_to_index[frozenset(n_simplex[p_simplex])] for p_simplex in p_simplices]
        bg_index = list(zip(p_indices, bg))
        bg_index.sort(key=lambda x: x[0])
        for i, (p_index1, differential1) in enumerate(bg_index):
            raised_differential1 = metric.dot(differential1.conj().T)
            star[p_index1, p_index1] = det(differential1.dot(raised_differential1))
            for j, (p_index2, differential2) in enumerate(bg_index[i + 1:]):
                star[p_index1, p_index2] = (-1) ** (j + 1) * det(differential2.dot(raised_differential1))
                
        return star