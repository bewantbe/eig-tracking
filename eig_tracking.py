#!/usr/bin/env python3

#import autograd.numpy as np
import numpy as np
from numpy import linspace, ones, abs
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
n_steps = 1000
np.random.seed(3264632)
A = np.random.rand(n,n)-0.5
B = np.random.rand(n,n)-0.5

A = np.diag(np.linspace(0.3,1.3,n))
B = 0.3*(np.random.rand(n,n)-0.5)

egval, egvec = np.linalg.eig(A)

id_sort = np.argsort(abs(egval))
vec_dir_ref = (egval[id_sort], egvec[:, id_sort])

s_vecs = np.zeros((n_steps, n, n), dtype='complex128')
s_vals = np.zeros((n_steps, n), dtype='complex128')

for k in range(n_steps):
    h = A + k/(n_steps-1) * B
    egval, egvec = np.linalg.eig(h)
    #eig_pick, vec_dir_ref = egval, egvec
    vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
    s_vals[k, :] = vec_dir_ref[0]
    s_vecs[k, :, :] = vec_dir_ref[1]

f_vandr = lambda lm, t: np.array([[lm*lm, lm, t, 1]]).T
f_vandr_l = lambda lm, t: np.array([[lm*lm, lm, t*lm, t, 1]]).T
svd_coplane_thres = 2e-4
repeated_root_thres = 0.05
crossing_root_test_thres = 0.2

s_cross = []
# find potential repeated eigenvalues and their location
for k in range(1,n_steps):
    # find close pairs of eigenvalues
    diff = abs(s_vals[k, :][:,np.newaxis] - s_vals[k, :][np.newaxis,:])
    id_so_diff = np.argsort(diff.flatten())
    n_close = np.sum(diff.flatten() < crossing_root_test_thres)
    for id_pick_close in range(n, n_close):
        ij1d = id_so_diff[id_pick_close]
        i = ij1d // n
        j = ij1d % n
        if i <= j:
            continue
        # assume pair lambda_i and lambda_j crossed
        t1 = (k-1)/(n_steps-1)
        t2 = k/(n_steps-1)
        l1t1 = s_vals[k-1,i]
        l2t1 = s_vals[k-1,j]
        l1t2 = s_vals[k,i]
        l2t2 = s_vals[k,j]
        # assume case 1, solve lambda_0 and t_0
        # (lambda - lambda_0)^2 + b_1 (t-t0) = 0
        V = np.hstack([f_vandr(l1t1, t1), f_vandr(l2t1, t1),
                       f_vandr(l1t2, t2), f_vandr(l2t2, t2)])
        #print('V.shape = ', V.shape)
        u,s,vh = np.linalg.svd(V)
        umin = u[:, np.argmin(s)]
        umin = umin / umin[0]  # the d1, d2, d3
        l0 = -umin[1]/2
        b1 = umin[2]
        t0 = (l0*l0 - umin[3]) / b1
        #if np.imag(t0) > 1e-9:
        #    print('What?? t0=', t0)
        t0 = np.real(t0)
        if not (t1 <= t0 and t0 <= t2 and s[-1] < svd_coplane_thres):
            continue
        h = A + t0 * B
        egval, egvec = np.linalg.eig(h)
        megval = (l1t1+l2t1+l1t2+l2t2)/4
        #print('::', abs(egval - megval))
        #print('::', np.argsort(abs(egval - megval)))
        l1t0, l2t0 = egval[np.argsort(abs(egval - megval))[0:2]]
        if abs(l1t0 - l2t0) > repeated_root_thres:
            continue
        print('Potential crossing k=%d, smin = %.2g:\n  t1,t0,t2=%.3f, %.3f, %.3f\n' % \
                (k, s[-1], t1, t0, t2), \
                '  lt1 =', l1t1, ', ', l2t1, '\n', \
                '  lt0 =', l1t0, ', ', l2t0, '\n', \
                '  lt0_guessed =', l0, '\n', \
                '  lt2 =', l1t2, ', ', l2t2)
        # Refinement
        V = np.hstack([f_vandr_l(l1t1, t1), f_vandr_l(l2t1, t1),
                       f_vandr_l(l1t0, t0), f_vandr_l(l2t0, t0),
                       f_vandr_l(l1t2, t2), f_vandr_l(l2t2, t2)])
        u_l, s_l, vh_l = np.linalg.svd(V)
        d = u_l[:, -1]
        lm_s = np.roots([d[0]*d[2], 2*d[0]*d[3], d[1]*d[3]-d[2]*d[4]])
        t0_s = -(2*d[0]*lm_s + d[1]) / d[2]   # could be unstable due to small u_l[2]
        # picking a solution
        id_s = [j for j in range(2) \
                  if abs(np.imag(t0_s[j]))<1e-10 and \
                     t1 <= np.real(t0_s[j]) and np.real(t0_s[j]) <= t2]
        print('  Refinement: t0_s =', t0_s, '  lm_s =', lm_s)
        if len(id_s) == 0:
            print("Didn't pass refinement test.")
            s_cross.append((t0, np.nan))  # make a refinement but do not plot
            continue
        if len(id_s) == 2:
            print("Multiple solution?!!")
        s_cross.append((np.real(t0_s[id_s[0]]), lm_s[id_s[0]]))

egval, egvec = np.linalg.eig(A)

id_sort = np.argsort(abs(egval))
vec_dir_ref = (egval[id_sort], egvec[:, id_sort])

s_vals_l = np.zeros((n_steps+len(s_cross), n), dtype='complex128')
s_vecs_l = np.zeros((n_steps+len(s_cross), n, n), dtype='complex128')
tt = linspace(0, 1, n_steps)
tt_l = np.zeros(n_steps+len(s_cross))

# generate eigen list including crossing points
k = 0
kk = 0
for idc in range(len(s_cross)):
    while k < (n_steps-1) * s_cross[idc][0]:
        egval = s_vals[k, :]
        egvec = s_vecs[k, :]
        vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
        s_vals_l[kk, :] = vec_dir_ref[0]
        s_vecs_l[kk, :, :] = vec_dir_ref[1]
        tt_l[kk] = tt[k]
        k += 1
        kk += 1
    h = A + s_cross[idc][0] * B
    egval, egvec = np.linalg.eig(h)
    vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
    s_vals_l[kk, :] = vec_dir_ref[0]
    s_vecs_l[kk, :, :] = vec_dir_ref[1]
    tt_l[kk] = s_cross[idc][0]
    kk += 1

while k < n_steps:
    egval = s_vals[k, :]
    egvec = s_vecs[k, :]
    vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
    s_vals_l[kk, :] = vec_dir_ref[0]
    s_vecs_l[kk, :, :] = vec_dir_ref[1]
    tt_l[kk] = tt[k]
    k += 1
    kk += 1

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
  ax.plot3D(tt_l, s_vals_l.real[:,ie], s_vals_l.imag[:,ie])

# mark crossing points
t_c = [t[0] for t in s_cross]
l_c = [t[1] for t in s_cross]
ax.scatter3D(t_c, np.real(l_c), np.imag(l_c), marker='o')

ax.set_xlabel('t')
ax.set_ylabel('eigval-real')
ax.set_zlabel('eigval-imag')

# only the real part
plt.figure(3)
for ie in range(n):
    plt.plot(tt_l, s_vals_l.real[:,ie])
    plt.xlabel('t')
    plt.ylabel('Re(eig)')

plt.show()

