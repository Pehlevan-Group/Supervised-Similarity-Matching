import numpy as np
import theano
import numpy as np
import scipy.sparse as sp
from theano import sparse
import itertools
import theano.tensor as T
from scipy.linalg import block_diag

'''
#More computationally efficient implementation O(r/sl*Al)
def place_on_grid(input_dim, stride): #0->Lth layer
	
	#param input_dim: 0th layer
	#param stride: stride def rel to L0
	#return: framelist,
	
	framelist=[] #List of lists. lth layer is a list with the x coords of the points in the layer and the y coords of the points in that layer
	x_indices, y_indices = np.meshgrid(np.arange(input_dim), np.arange(input_dim), indexing='ij')
	framelist.append([x_indices, y_indices])
	for s in stride:
		coords=np.arange(0, input_dim, step=s)
		x_indices, y_indices=np.meshgrid(coords, coords, indexing='ij')
		framelist.append([x_indices, y_indices])
	return framelist

def get_circumsquare_mask_for_layer(radii, stride):
	
	#param dl_list: list of d's for lth layer
	#param radius: Euclidean radius, 1->L (L+1th is FC)
	#return: the kernel size (smaller tiled matrix dim) kernel_size=4 implies each neuron receives stim from a 4x4xnps region in l-1
	
	square_size=[2*np.int(radl/sl) for radl, sl in zip(radii, stride)] #side of the circumsquare

	def get_kernelshape_for_layer(prev_layer_size, kernel_size): #assuming for
		S_xy=np.zeros(prev_layer_size, prev_layer_size) #d_l-1*d_l-1
		mask=np.ones(kernel_size, kernel_size)
'''

'''def check_within_circle(coord1, coord2, radius):
	if (((coord1[0]-coord2[0])**2.0 + (coord1[1]-coord2[1])**2.0)<radius):
		return 1
	else:
		return 0'''


def get_layerdims_from_stride(stride, dim0):  # Layer grid dimension for 0->L+1
	L = len(stride) - 1
	layer_dims = np.zeros(L + 2)  # Corresponding size=dim*di
	layer_dims[0] = dim0
	for l in range(1, L + 1):
		layer_dims[l] = dim0 / stride[l]
	layer_dims[-1] = 10
	return np.array(layer_dims, dtype=np.int32)


def get_S_for_layers(d_prev, d_cur, s_prev, s_cur, nps_prev, nps_cur, radius):
	'''
	for layers 1->L
	:param d_prev, d_cur: Side of previous and current layer, range of x, y coords
	:param s_prev, s_cur: Stride of prev and current layer, ind(l)*sl = eucl_l, 1 for first layer
	:param radius: rl, defined wrt previous layer
	:return:
	'''
	Vprev = np.power(d_prev, 2) * nps_prev
	Vcur = np.power(d_cur, 2) * nps_cur

	ind_lprev = np.arange(Vprev)
	ind_lcur = np.arange(Vcur)
	euclid_prev = [(ind_lprev / (nps_prev * d_prev)) * s_prev,
				   ((ind_lprev / nps_prev)%d_prev) * s_prev]  # (i, j)th index*stride for l-1. y changes faster
	euclid_cur = [(ind_lcur / (nps_cur * d_cur)) * s_cur, ((ind_lcur / nps_cur)% d_cur) * s_cur]
	distmat = np.subtract.outer(euclid_prev[0], euclid_cur[0]) ** 2.0 + np.subtract.outer(euclid_prev[1],
																						  euclid_cur[1]) ** 2.0
	S = np.array(distmat <= (radius ** 2.0), dtype=theano.config.floatX)
	return S

def get_S_lateral_for_layers(d_l, nps_l):
	lat_connex=np.array(np.ones((nps_l, nps_l)), dtype=theano.config.floatX)
	bd_list=[lat_connex]*((d_l)**2)
	S_lat=block_diag(*bd_list)
	return S_lat


def test_S():
	Smatrix = get_S_for_layers(64, 32, 1, 2, 1, 3, 8)
	print 'fin'
	return

def get_structured_matices(stride, nps, euclid_radii, input_dim):
	'''
	Example:
	nps_const = [3, 3, 3], len=L
	stride = [1, 2, 4, 8]  # 0->L, Must always start with 1, len=L+1
	euclid_radii = [8, 12, 24], len=L
	input_dim = 64
	RETURNS: List of structured matrices for W and L
	'''
	L = len(stride) - 1

	dlist = get_layerdims_from_stride(stride, input_dim)  # 0->L+1
	stride_input = stride[:(-1)]
	stride_output = stride[1:]
	Slatmatrices = [get_S_lateral_for_layers(d_l, nps_l) for d_l, nps_l in zip(dlist[1:(-1)], nps)]
	nps = [1] + nps
	nps_inp, nps_out = nps[:(-1)], nps[1:]
	Smatrices = [get_S_for_layers(d_prev, d_cur, s_prev, s_cur, nps_prev, nps_cur, radius) for
				 d_prev, d_cur, s_prev, s_cur, nps_prev, nps_cur, radius in
				 zip(dlist[:-2], dlist[1:(L + 1)], stride_input, stride_output, nps_inp, nps_out,
					 euclid_radii)]  # L matrices
	return Smatrices, Slatmatrices #not sparse

#SPARSE FUNCS
def sparse_batched_dot(x, y):
	#Returns a dense matrix
	return sparse.sp_sum(sparse.basic.mul(sparse.csc_from_dense(x), y), sparse_grad=True, axis=1)


if __name__ == '__main__':
	'''
	nps_const = [2] #len = L
	stride = [1, 2]  # 0->L, Must always start with 1, len=L+1
	euclid_radii = [2] #len = L
	input_dim = 8
	Sfwd, Slat=get_structured_matices(stride, nps_const, euclid_radii, input_dim)
	print 4'''




'''
Next steps:
Convert to list of sparse theano matrices of S
Multiply with W to get augmented W
GET L MATRIX (also super sparse)
proceed as usual
ENSURE THAT SPARSE ELEMENTS ARE NOT UPDATED SO STRUCTURED GRADIENTS EVERYWHERE
'''
# grid_coords=place_on_grid(28, stride)
# kermat_sizes=obtain_kernel_size_from_radius(stride, euclid_radii)
# print kermat_sizes
