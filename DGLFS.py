# -*- coding:utf-8 -*-
"""
作者：彭诚
日期：2025年03月07日
"""
import numpy as np
from numpy.linalg import norm, LinAlgError
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import solve, solve_sylvester
from scipy.linalg import cho_factor, cho_solve
from DPC import select_dc, get_density, get_deltas, find_centers_K

eps = np.finfo(np.float64).eps
np.random.seed(1)
np.set_printoptions(threshold=10000, linewidth=2000)


def relax_l21(W):
	'''

    计算对矩阵计算L2，1范数的松弛矩阵
    :param W:
    :return:
    '''
	dv, q = W.shape
	C = np.zeros((dv, dv))  # 初始化对角矩阵C
	# 计算对角元素
	for i in range(dv):
		C[i, i] = 1 / (0.5 * np.sqrt(np.sum(W[i] * W[i]) + eps))
	return C


def cluster_center(M, dist, n_clusters):
	"""

    计算聚类中心 DPC
    param: M: np.ndarray, (n, d)
    param: dist: 距离矩阵
    n_clusters: int,
    """
	dc = select_dc(dist)
	rho = get_density(dist, dc, method="Gaussion")
	deltas, _ = get_deltas(dist, rho)
	centers = find_centers_K(rho, deltas, n_clusters)
	result_matrix = M[centers, :]

	return result_matrix


def dist_calculate(X, Y):
	"""

    计算矩阵距离
    param: X: np.ndarray, (n, d)
    param: Y: np.ndarray, (m, d)
    """
	n, d = X.shape
	m, d = Y.shape
	result_matrix = np.zeros((n, m), dtype=float)

	for i in range(n):
		for j in range(m):
			result_matrix[i][j] = np.linalg.norm(X[i] - Y[j]) ** 2 + eps

	return result_matrix


def update_membershipValue(D, m):
	'''

	更新隶属度矩阵
    :param D: 距离矩阵
    :param m: 隶属度矩阵的指数
    '''

	D_p = 1.0 / (D ** (1.0 / (m - 1)))
	result_matrix = D_p / np.sum(D_p, axis=1)[:, np.newaxis]
	return result_matrix


def cholesky_solve(A, B) -> np.ndarray:
	try:
		c, lower = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
		return cho_solve((c, lower), B, overwrite_b=True, check_finite=False)
	except LinAlgError:
		return solve(A, B, overwrite_b=True, check_finite=False)


def IGDF_PML(X, Y, lambda1, lambda2, lambda4, lambda3):
	'''

    :param X: 特征矩阵， n*d
    :param Y: 标签矩阵， n*q
    :param lambda1:  local disambiguation 的系数
    :param lambda2:  global disambiguation 的系数
    :param lambda3: ||H||_F^2 的系数
    :param lambda4: ||AtB||_{2,1} 的系数
    :return: f_idx, G, obj_save, len(obj_save)
    '''

	n, d = X.shape
	n, q = Y.shape

	maxIter = 200
	theta = 0.4
	m = 2
	c = 50

	XXt = X @ X.T
	G = np.random.uniform(0, 1, size=(n, q))
	G = G * Y
	H = np.eye(n)
	HX = X

	dist_x = squareform(pdist(X, 'euclidean'))
	dist_y = squareform(pdist(Y, 'euclidean'))
	A = cluster_center(X, dist_x, n_clusters=c)
	B = cluster_center(Y, dist_y, n_clusters=c)
	AtB = A.T @ B
	C = relax_l21(AtB)

	Dx = dist_calculate(X, A)
	Dg = dist_calculate(G, B)

	F = update_membershipValue(Dx + Dg, m)
	F_m = F ** m
	D_c = np.diag(F_m.sum(axis=0))
	D_n = np.diag(F_m.sum(axis=1))
	I_n = np.eye(n)
	I_c = np.eye(c)

	iteration = 0
	obj_save = []
	obj = (norm(HX @ AtB - G, 'fro') ** 2
	       + lambda4 * np.sum(np.sqrt(np.sum(AtB * AtB, 1))) + lambda3 * norm(H, 'fro') ** 2
	       + lambda1 * (np.sum(F_m * Dx) + np.sum(F_m * Dg))
	       + lambda2 * (norm(HX - X, 'fro') ** 2 + norm(H @ G - G, 'fro') ** 2))
	obj_save.append(float(obj))
	print('the object value in iter {} is {}\n'.format(iteration, obj_save[iteration]))

	iteration = 1
	while iteration < maxIter:
		# Update G
		G = cholesky_solve(I_n + lambda1 * D_n + lambda2 * (H - I_n).T @ (H - I_n),
		                   HX @ AtB + lambda1 * F_m @ B)
		G = np.clip(G, np.zeros_like(G), Y)

		# Update B
		B = cholesky_solve(A @ HX.T @ HX @ A.T + lambda1 * D_c + lambda4 * A @ C @ A.T,
		                   A @ HX.T @ G + lambda1 * F_m.T @ G)
		BBT = B @ B.T + 1e-8 * I_c

		# Update A
		M = lambda1 * solve(BBT, D_c)
		N = HX.T @ HX + lambda4 * C
		Q = solve(BBT, (B @ G.T @ HX + lambda1 * F_m.T @ X))
		A = solve_sylvester(M, N, Q)

		AtB = A.T @ B
		C = relax_l21(AtB)

		# Update H
		H_1 = lambda2 * (G @ G.T + XXt)
		H_2 = AtB.T @ X.T
		H = cholesky_solve((H_2.T @ H_2 + H_1 + lambda3 * I_n).T,
		                   (G @ H_2 + H_1).T).T
		HX = H @ X

		# Update F
		Dx = dist_calculate(X, A)
		Dg = dist_calculate(G, B)
		F = update_membershipValue(Dx + Dg, m)
		F_m = F ** m
		D_c = np.diag(F_m.sum(axis=0))
		D_n = np.diag(F_m.sum(axis=1))

		# 判断退出条件
		obj = (norm(HX @ AtB - G, 'fro') ** 2
		       + lambda4 * np.sum(np.sqrt(np.sum(AtB * AtB, 1))) + lambda3 * norm(H, 'fro') ** 2
		       + lambda1 * (np.sum(F_m * Dx) + np.sum(F_m * Dg))
		       + lambda2 * (norm(HX - X, 'fro') ** 2 + norm(H @ G - G, 'fro') ** 2))
		obj_save.append(float(obj))
		print('the object value in iter {} is {}\n'.format(iteration, obj_save[iteration]))

		if (iteration > 2) and (abs(obj_save[iteration] - obj_save[iteration - 1]) / abs(obj_save[iteration]) < 0.001):
			print('break')
			break

		# Next iteration
		iteration += 1

	w_2 = norm(AtB, ord=2, axis=1)
	f_idx = np.argsort(-w_2).tolist()

	G_min = G.min(axis=1, keepdims=True)
	G_max = G.max(axis=1, keepdims=True)
	G = (G - G_min) / (G_max - G_min)
	G[np.isnan(G)] = 0

	G[G >= theta] = 1
	G[G < theta] = 0
	G = G.astype(int)

	return f_idx, G, obj_save, len(obj_save)
