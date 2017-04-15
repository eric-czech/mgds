
import edward as ed
import tensorflow as tf
import numpy as np
import pandas as pd
from edward.models import Normal, Gamma, Laplace, NormalWithSoftplusSigma, GammaWithSoftplusAlphaBeta
from edward.models import MultivariateNormalFull, MultivariateNormalDiag, MultivariateNormalDiagPlusVDVT, MultivariateNormalCholesky
from edward.models import PointMass, InverseGamma, Uniform, StudentT
from ml.edward import models as ed_models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_distances

def compute_kernel(X, Xs=None, gamma=None):
    return rbf_kernel(X, Y=Xs, gamma=gamma)


class MTKLModel(ed_models.BayesianModel):

    def __init__(self, R, X_rppa, Y_rppa, gamma=None, print_kernel_stats=True):
        self.initialize()
        self.gamma = gamma
        self.R = R
        self.X_rppa = X_rppa
        self.Y_rppa = Y_rppa
        self.print_kernel_stats = print_kernel_stats

    def initialize(self):
        self.x_rppa_scaler = StandardScaler()
        self.x_drug_scaler = StandardScaler()
        self.y_rppa_scaler = StandardScaler()
        self.y_drug_scaler = StandardScaler()

    def add_rppa_model(self, K_rppa, N, R, T, lv, tm):

        A_sig = 1.
        A = Normal(mu=tf.zeros([N, R]), sigma=A_sig*tf.ones([N, R]))
        qA = PointMass(params=tf.Variable(tf.random_normal([N, R], stddev=.1), name='A_mu'))

        # H is now N x R
        H = tf.matmul(K_rppa, A)

        W_sig = 1.
        W = Normal(mu=tf.zeros([R, T]), sigma=W_sig*tf.ones([R, T]))
        qW = PointMass(params=tf.Variable(tf.random_normal([R, T], stddev=.1), name='W_mu'))

        B_sig = 1.
        B = Normal(mu=tf.zeros([1, T]), sigma=B_sig*tf.ones([1, T]))
        qB = PointMass(params=tf.Variable(tf.random_normal([1, T], stddev=.1), name='B_mu'))

        # Y_rppa = Normal(mu=tf.matmul(H, W) + B, sigma=.3 * tf.ones([N, T]))
        Y_rppa = StudentT(df=3*tf.ones([N, T]), mu=tf.matmul(H, W) + B, sigma=.1 * tf.ones([N, T]))

        Y_rppa_pred = tf.matmul(tf.matmul(K_rppa, qA.params), qW.params) + qB.params

        lv.update({B:qB, A:qA, W:qW})
        tm.update({
            'qB': qB.params, 'B': B,
            'qW': qW.params, 'W': W,
            'qA': qA.params, 'A': A,
            'Y_rppa': Y_rppa,
            'Y_rppa_pred': Y_rppa_pred
        })

    def add_drug_model(self, K_drug, N, T1, T2, lv, tm):

        qY_drug_rppa = tf.matmul(tf.matmul(K_drug, tm['qA']), tm['qW']) + tm['qB']
        Y_drug_rppa = tf.matmul(tf.matmul(K_drug, tm['A']), tm['W']) + tm['B']

        G_sig = 1.
        # G = Normal(mu=tf.zeros([T1, T2]), sigma=G_sig*tf.ones([T1, T2]))
        G = Laplace(loc=tf.zeros([T1, T2]), scale=G_sig*tf.ones([T1, T2]))
        qG = PointMass(params=tf.Variable(tf.random_normal([T1, T2], stddev=.1), name='G_mu'))

        I_sig = 1.
        # I = Normal(mu=tf.zeros([1, T2]), sigma=I_sig*tf.ones([1, T2]))
        I = Laplace(loc=tf.zeros([1, T2]), scale=I_sig*tf.ones([1, T2]))
        qI = PointMass(params=tf.Variable(tf.random_normal([1, T2], stddev=.1), name='I_mu'))

        # Y_drug = Normal(mu=tf.matmul(qY_drug_rppa, G) + I, sigma=.1 * tf.ones([N, T2]))
        # Y_drug = Normal(mu=tf.matmul(Y_drug_rppa, G) + I, sigma=.3 * tf.ones([N, T2]))
        Y_drug = StudentT(df=3*tf.ones([N, T2]), mu=tf.matmul(Y_drug_rppa, G) + I, sigma=.1 * tf.ones([N, T2]))

        Y_drug_pred = tf.matmul(qY_drug_rppa, qG.params) + qI.params

        lv.update({G:qG, I:qI})
        tm.update({
            'qG': qG.params,
            'qI': qI.params,
            'Y_drug': Y_drug,
            'qY_drug_rppa': qY_drug_rppa,
            'Y_drug_pred': Y_drug_pred
        })

    def inference_args(self, data):
        self.initialize()

        dX_rppa, dY_rppa = self.X_rppa, self.Y_rppa
        dX_drug, dY_drug = data['X'], data['Y']

        self.x_rppa_scaler.fit(dX_rppa)
        self.x_drug_scaler.fit(dX_drug)
        dY_rppa = self.y_rppa_scaler.fit_transform(dY_rppa)
        dY_drug = self.y_drug_scaler.fit_transform(dY_drug)

        dX_rppa = self.x_rppa_scaler.transform(dX_rppa)
        dX_drug = self.x_drug_scaler.transform(dX_drug)

        dK_rppa = compute_kernel(dX_rppa, gamma=self.gamma)
        dK_drug = compute_kernel(dX_drug, dX_rppa, gamma=self.gamma)

        def kernel_stats(k):
            return pd.Series(k.ravel()).describe(percentiles=[.01, .1, .25, .50, .75, .90, .99])

        if self.print_kernel_stats:
            print('RPPA Kernel Stats: {}'.format(kernel_stats(dK_rppa)))
            print('Drug Kernel Stats: {}'.format(kernel_stats(dK_drug)))

        K_rppa = tf.constant(dK_rppa, dtype=tf.float32)
        K_drug = tf.constant(dK_drug, dtype=tf.float32)

        N1, P1 = dX_rppa.shape
        N2, P2 = dX_drug.shape
        T1, T2 = dY_rppa.shape[1], dY_drug.shape[1]
        R = self.R

        # print({'N_rppa': N1, 'N_drug': N2, 'P_rppa': P1, 'P_drug': P2, 'T_rppa': T1, 'T_drug': T2, 'R': R})

        lv, tm = {}, {}

        tm['dX_rppa'] = dX_rppa

        self.add_rppa_model(K_rppa, N1, R, T1, lv, tm)

        self.add_drug_model(K_drug, N2, T1, T2, lv, tm)

        # for k, v in tm.items():
        #     try:
        #         print(k, v.get_shape())
        #     except:
        #         pass

        mse_rppa = tf.reduce_mean(tf.square(tm['Y_rppa_pred'] - tf.constant(dY_rppa, dtype=tf.float32)), axis=0)
        mse_drug = tf.reduce_mean(tf.square(tm['Y_drug_pred'] - tf.constant(dY_drug, dtype=tf.float32)), axis=0)

        tf.summary.scalar('mse_rppa', tf.reduce_mean(mse_rppa))
        tf.summary.scalar('mse_drug', tf.reduce_mean(mse_drug))

        def input_fn(d):
            return {tm['Y_rppa']: dY_rppa, tm['Y_drug']: dY_drug}

        return input_fn, lv, tm

    def criticism_args(self, sess, tm):
        dW = sess.run(tm['qW'])
        dB = sess.run(tm['qB'])
        dA = sess.run(tm['qA'])
        dG = sess.run(tm['qG'])
        dI = sess.run(tm['qI'])

        def drug_pred_fn(X):
            dK_drug = compute_kernel(self.x_drug_scaler.transform(X), tm['dX_rppa'], gamma=self.gamma)
            Y_rppa = np.matmul(np.matmul(dK_drug, dA), dW) + dB
            Y_drug = np.matmul(Y_rppa, dG) + dI
            return self.y_drug_scaler.inverse_transform(Y_drug)

        def drug_pred_train_fn():
            return self.y_drug_scaler.inverse_transform(sess.run(tm['Y_drug_pred']))

        def rppa_pred_train_fn():
            return self.y_rppa_scaler.inverse_transform(sess.run(tm['Y_rppa_pred']))

        return drug_pred_fn, drug_pred_train_fn, rppa_pred_train_fn


#model = MTKLModel(50, gamma=None)