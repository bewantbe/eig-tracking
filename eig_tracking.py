#!/usr/bin/env python3

#import autograd.numpy as np
import numpy as np
from numpy import linspace, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Return eigen vector and value for desired direction.
# @param egval: eigen values list.
# @param egvec: each column is an eigen vector, normalized to 2-norm == 1.
# @param direction_ref: reference direction, can be a vector or column vectors.
# @param mode:
# @param ord: to select the `ord`-th close direction. For switching direction.
def pick_eigen_direction(egval, egvec, direction_ref, mode, ord=0):
    n = egvec.shape[-1]                # matrix dimension
    if mode == 'smallest':
        id_egmin = np.argmin(np.abs(egval))
        zero_g_direction = egvec[:, id_egmin]
        if np.dot(zero_g_direction, direction_ref) < 0:
            zero_g_direction = -zero_g_direction
        return egval[id_egmin], zero_g_direction
    elif mode == 'continue':
        # pick the direction that matches the previous one
        if direction_ref.ndim==1:
            direction_ref = direction_ref[:, np.newaxis]
        n_pick = direction_ref.shape[1]    # number of track
        vec_pick = np.zeros_like(direction_ref)
        val_pick = np.zeros_like(egval)
        similarity = np.dot(egvec.conj().T, direction_ref)
        for id_v in range(n_pick):
            # id_pick = np.argmin(np.abs(np.abs(similarity[:, id_v])-1))
            id_pick = np.argsort(np.abs(np.abs(similarity[:, id_v])-1))[ord]
            if similarity[id_pick, id_v] > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            val_pick[id_v] = egval[id_pick]
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    elif mode == 'close-egval':
        # direction_ref should be pair (egval, egvec)
        old_egval = direction_ref[0]
        old_egvec = direction_ref[1]
        if len(old_egvec.shape) == 1:
            old_egvec = old_egvec[:, np.newaxis]
        egdist = np.abs(old_egval[:,np.newaxis] - egval[np.newaxis, :])
        # Greedy pick algo 1
        # 1. loop for columns of distance matrix
        # 2.    pick most close pair of eigenvalue and old-eigenvalue.
        # 3.    remove corresponding row in the distance matrix.
        # 4. go to 1.
        # Greedy pick algo 2
        # 1. pick most close pair of eigenvalue and old-eigenvalue.
        # 2. remove corresponding column and row in the distance matrix.
        # 3. go to 1.
        # Use algo 1.
        #plt.matshow(egdist)
        n_pick = old_egvec.shape[1]
        mask = np.arange(n_pick)
        vec_pick = np.zeros_like(old_egvec)
        val_pick = np.zeros_like(egval)
        #print('old_eigval=\n', old_egval[:,np.newaxis])
        #print('new eigval=\n', egval[:,np.newaxis])
        for id_v in range(n_pick):
            id_pick_masked = np.argmin(egdist[id_v, mask])
            #print('mask=',mask)
            id_pick = mask[id_pick_masked]
            #print('id_pick=',id_pick, '  eigval=', egval[id_pick])
            val_pick[id_v] = egval[id_pick]
            # might try: sign = np.exp(np.angle(...)*1j)
            if np.angle(np.vdot(egvec[:,id_pick], old_egvec[:, id_v])) > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            mask = np.delete(mask, id_pick_masked)
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    else:
        raise ValueError()

n = 10
n_steps = 500
np.random.seed(3264632)
A = np.random.rand(n,n)-0.5
B = np.random.rand(n,n)-0.5

egval, egvec = np.linalg.eig(A)

id_sort = np.argsort(abs(egval))
vec_dir_ref = (egval[id_sort], egvec[:, id_sort])

s_vecs = np.zeros((n_steps, n, n), dtype='complex128')
s_vals = np.zeros((n_steps, n), dtype='complex128')

for k in range(n_steps):
    h = A + 0.99*k/n_steps * B
    egval, egvec = np.linalg.eig(h)
    #eig_pick, vec_dir_ref = egval, egvec
    vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
    s_vals[k, :] = vec_dir_ref[0]
    s_vecs[k, :, :] = vec_dir_ref[1]

#plt.figure(34)
#plt.plot(s_vals.real)
#plt.ylabel('eigval-real')

#plt.figure(35)
#plt.plot(s_vals.imag)
#plt.ylabel('eigval-imag')

tt = linspace(0, 1, n_steps)

fig = plt.figure(234)
ax = fig.gca(projection='3d')
for ie in range(n):
  ax.plot3D(tt, s_vals.real[:,ie], s_vals.imag[:,ie])

ax.set_xlabel('t')
ax.set_ylabel('eigval-real')
ax.set_zlabel('eigval-imag')

plt.show()

