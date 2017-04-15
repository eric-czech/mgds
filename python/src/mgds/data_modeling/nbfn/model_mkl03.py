
import edward as ed
import tensorflow as tf
import numpy as np
import pandas as pd
from edward.models import Normal, Gamma, Laplace, NormalWithSoftplusSigma, GammaWithSoftplusAlphaBeta
from edward.models import MultivariateNormalFull, MultivariateNormalDiag, MultivariateNormalDiagPlusVDVT, MultivariateNormalCholesky
from edward.models import PointMass, InverseGamma, Uniform, StudentT, Dirichlet, Categorical
from ml.edward import models as ed_models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_distances


def compute_kernel(X, Xs=None, gamma=None):
    return rbf_kernel(X, Y=Xs, gamma=gamma)


class MTKLModel(ed_models.BayesianModel):

    def __init__(self, group_names, gamma=None, print_kernel_stats=True):
        self.gamma = gamma
        self.print_kernel_stats = print_kernel_stats
        self.group_names = group_names
        self.n_group = len(group_names)

    def _std(self, Y):
        devs_squared = tf.square(Y - tf.reduce_mean(Y, axis=0))
        return tf.sqrt(tf.reduce_mean(devs_squared, axis=0))

    def _pearson(self, Y1, Y2):
        p_top = tf.reduce_mean((Y1 - tf.reduce_mean(Y1, axis=0)) * (Y2 - tf.reduce_mean(Y2, axis=0)), axis=0)
        p_bottom = self._std(Y1) * self._std(Y2)
        return p_top / p_bottom

    def inference_args(self, data):

        input_map = {}
        latent_map = {}
        tensor_map = {}
        param_ct = {}

        dXs = data['X']
        dYs = data['Y']

        tensor_map['dXs'] = dXs
        tensor_map['dYs'] = dYs

        assert len(dXs) == len(dYs) == self.n_group

        Ng = self.n_group
        N = np.sum([len(d) for d in dXs])
        tensor_map['N'] = N

        A = Dirichlet(alpha=5.*tf.ones(Ng))
        # qA = Dirichlet(alpha=tf.nn.softplus(tf.Variable(tf.random_normal([n_src]))))
        #qA = PointMass(params=tf.nn.softplus(tf.Variable(tf.random_normal([Ng]))))
        qA = PointMass(params=tf.nn.softplus(tf.Variable(tf.ones(Ng, dtype=tf.float32))))
        #qA = PointMass(params=tf.nn.softplus(tf.ones(Ng, dtype=tf.float32)))
        latent_map[A] = qA
        param_ct['qA'] = Ng
        tensor_map['qA'] = qA
        # tf.summary.histogram('qA', qA.params)
        # tf.summary.histogram('A', A)
        qAp = qA.params / tf.reduce_sum(qA.params)

        for i in range(Ng):
            group_name = self.group_names[i]
            Ni, Ti = dYs[i].shape

            tf.summary.scalar('p_{}'.format(group_name), tf.gather(qAp, i))

            # Initialize observation weight matrix for group
            # Wi = Normal(mu=tf.zeros([N, Ti]), sigma=.3*tf.ones([N, Ti]))
            Wi = Laplace(loc=tf.zeros([N, Ti]), scale=1.*tf.ones([N, Ti]))
            qWi = PointMass(params=tf.Variable(tf.random_normal([N, Ti], stddev=.1), name='W{}'.format(i)))
            latent_map[Wi] = qWi
            tensor_map['qW{}'.format(group_name)] = qWi.params
            param_ct['qW{}'.format(group_name)] = np.prod(qWi.get_shape().as_list())
            tf.summary.histogram('qW{}'.format(group_name), qWi.params)

            # Initialize intercept for group
            # Bi = Normal(mu=tf.zeros([1, Ti]), sigma=1.*tf.ones([1, Ti]))
            Bi = Laplace(loc=tf.zeros([1, Ti]), scale=1.*tf.ones([1, Ti]))
            qBi = PointMass(params=tf.Variable(tf.random_normal([1, Ti], stddev=.1), name='B{}'.format(i)))
            latent_map[Bi] = qBi
            tensor_map['qB{}'.format(group_name)] = qBi.params
            param_ct['qB{}'.format(group_name)] = np.prod(qBi.get_shape().as_list())
            tf.summary.histogram('qB{}'.format(group_name), qBi.params)

            # Initialize stacked kernel matrix for group
            Ki = []
            for j in range(Ng):
                Xi, Xj = dXs[i], dXs[j]
                Nj = len(Xj)
                Kij = compute_kernel(Xj, Xi, gamma=self.gamma)
                assert Kij.shape == (Nj, Ni)

                # Get scalar kernel weight for this group
                #aj = tf.gather(A, j)
                aj = tf.gather(qAp, j)
                #aj = 0. if i == j else 1.
                #aj = 0. if self.group_names[j] == 'ccle_v1' else 1.

                Ki.append(tf.cast(aj, tf.float32) * Kij)
                #Ki.append(tf.cast(Kij, dtype=tf.float32))

            # Stack kernels vertically (retaining column dimension) and then transpose to give Ni x N matrix
            Ki = tf.transpose(tf.concat(Ki, 0))
            assert Ki.get_shape().as_list() == [Ni, N]

            # Generate response prediction (Ni x Ti)
            Yi_mu = tf.matmul(Ki, Wi) + Bi
            assert Yi_mu.get_shape().as_list() == [Ni, Ti]

            Yi = StudentT(df=3.*tf.ones([Ni, Ti]), mu=Yi_mu, sigma=.5 * tf.ones([Ni, Ti]))
            #Yi = Normal(mu=Yi_mu, sigma=1. * tf.ones([Ni, Ti]))
            assert Yi.get_shape().as_list() == [Ni, Ti]

            qYi = tf.matmul(Ki, qWi.params) + qBi.params
            assert qYi.get_shape().as_list() == [Ni, Ti]

            # Bind prediction to data
            input_map[Yi] = dYs[i]
            tensor_map['qY{}'.format(group_name)] = qYi

            Yp, Yt = qYi, dYs[i]
            mse = tf.reduce_mean(tf.square(Yp - Yt), axis=0)
            tf.summary.scalar('mse_group_{}'.format(group_name), tf.reduce_mean(mse))
            pearson = self._pearson(Yp, tf.constant(Yt, dtype=tf.float32))
            tf.summary.scalar('pearson_{}'.format(group_name), tf.reduce_mean(pearson))


        print(pd.Series(param_ct))
        print(pd.Series(param_ct).sum())

        # if self.print_kernel_stats:
        #     print('RPPA Kernel Stats: {}'.format(kernel_stats(dK_rppa)))
        #     print('Drug Kernel Stats: {}'.format(kernel_stats(dK_drug)))

        def input_fn(d):
            return input_map

        return input_fn, latent_map, tensor_map

    def criticism_args(self, sess, tm):

        Ng = self.n_group
        N = tm['N']
        dXs = tm['dXs']
        dYs = tm['dYs']

        A = sess.run(tm['qA'])
        assert A.ndim == 1 and len(A) == Ng
        Ap = A / np.sum(A)

        def pred_fn_group(X, group_name):
            i = self.group_names.index(group_name)
            Nstar = X.shape[0]
            Ti = dYs[i].shape[1]

            W = sess.run(tm['qW{}'.format(group_name)])
            B = sess.run(tm['qB{}'.format(group_name)])

            Ki = []
            for j in range(Ng):
                Xj = dXs[j]
                Nj = dXs[j].shape[0]
                Kij = compute_kernel(Xj, X, gamma=self.gamma)
                assert Kij.shape == (Nj, Nstar)

                # Get scalar kernel weight for this group
                aj = Ap[j]
                # aj = 0. if self.group_names[j] == 'ccle_v1' else 1.
                # aj = 0. if i == j else 1.

                Ki.append(aj * Kij)

            # Stack vertically, retaining column dimension
            Ki = np.transpose(np.concatenate(Ki))
            assert Ki.shape == (Nstar, N)

            Y = np.matmul(Ki, W) + B
            assert Y.shape == (Nstar, Ti)

            return Y

        def pred_train_group(group_name):
            return sess.run(tm['qY{}'.format(group_name)])

        def dirichlet_params():
            A = sess.run(tm['qA'])
            Ap = A / np.sum(A)
            return A, Ap

        return pred_fn_group, dirichlet_params, pred_train_group
