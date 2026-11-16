"""seRNN experiment runner. Reuses the original notebook's classes (loaded from the copy
experiments/seRNN_demo.ipynb) without modifying them; only adds experiment factors on top:
regulariser type (none / l1 / dist / comm), regularisation strength, seed, distance_power, delay length.
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.spatial.distance, networkx as nx, bct
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.regularizers import Regularizer
from tensorflow.python.keras import backend

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'seRNN')
os.makedirs(OUT, exist_ok=True)

# --- load original definitions (maze generator, SE1, SE1_sWc, callback) from the notebook copy ---
_nb = json.load(open(os.path.join(HERE, 'seRNN_demo.ipynb'), encoding='utf8'))
_ns = globals()
for idx in (9, 15, 16, 18):
    exec(''.join(_nb['cells'][idx]['source']), _ns)


def structure_metrics(W, dist, nperm=100, seed=0):
    A = np.abs(W)
    thresh = np.quantile(A, 0.9)
    B = (A > thresh).astype(float)
    Bs = ((B + B.T) > 0).astype(float)
    np.fill_diagonal(Bs, 0)
    g = nx.from_numpy_array(Bs)
    comms = nx.community.louvain_communities(g, seed=seed)
    mod = nx.community.modularity(g, comms)
    clu = np.mean(bct.clustering_coef_bu(Bs))
    pth = bct.efficiency_bin(Bs)
    rng = np.random.default_rng(seed)
    cl, pt = [], []
    for _ in range(nperm):
        P = rng.random((100, 100)); P = (P + P.T) / 2
        Ap = (P > .7).astype(float); np.fill_diagonal(Ap, 0)
        cl.append(np.mean(bct.clustering_coef_bu(Ap))); pt.append(bct.efficiency_bin(Ap))
    smw = (clu / np.mean(cl)) / (np.mean(pt) / pth)
    iu = np.triu_indices(100, 1)
    corr = np.corrcoef(A[iu], dist[iu])[0, 1]
    edge_len = float((B * dist).sum() / max(B.sum(), 1))
    return dict(modularity=float(mod), smallworld=float(smw), wd_corr=float(corr),
                total_abs_w=float(A.sum()), mean_edge_len=edge_len, n_comms=len(comms))


def run(kind='comm', strength=0.5, seed=1, epochs=10, distance_power=1, delay=10, comms_factor=1, tag=None):
    name = tag or f'{kind}_s{strength}_seed{seed}_dp{distance_power}_delay{delay}'
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    np.random.seed(seed); tf.random.set_seed(seed + 1000); keras.backend.clear_session()
    gen = mazeGeneratorI(goal_presentation_steps=20, delay_steps=delay, choices_presentation_steps=20)
    tr = gen.construct_tf_data(5120, 128); te = gen.construct_tf_data(2560, 128)
    ex = next(iter(tr))
    if kind == 'none': regu = None
    elif kind == 'l1': regu = keras.regularizers.L1(strength)
    elif kind == 'dist': regu = SE1(se1=strength, distance_power=distance_power)
    elif kind == 'comm': regu = SE1_sWc(se1=strength, comms_factor=comms_factor, distance_power=distance_power)
    else: raise ValueError(kind)
    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(stddev=0.05),
        tf.keras.layers.SimpleRNN(100, activation='relu', recurrent_initializer='orthogonal',
                                  return_sequences=False, recurrent_regularizer=regu),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.build(input_shape=ex[0].shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    h = model.fit(tr, epochs=epochs, validation_data=te, callbacks=RNNWeightMatrixHistoryI(RNN_layer_number=1), verbose=0)
    W = np.array(h.history['RNN_Weight_Matrix'][-1])
    dist = SE1(se1=0.01, distance_power=1).distance_matrix  # reference distance (power 1) for all runs
    res = dict(name=name, kind=kind, strength=strength, seed=seed, epochs=epochs, distance_power=distance_power, delay=delay,
               train_acc=float(h.history['accuracy'][-1]), test_acc=float(h.history['val_accuracy'][-1]),
               acc_curve=[float(x) for x in h.history['val_accuracy']], seconds=time.time() - t0)
    res.update(structure_metrics(W, dist, seed=seed))
    np.save(os.path.join(OUT, name + '_W.npy'), W)
    json.dump(res, open(path, 'w'))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k in
                 ('test_acc', 'modularity', 'smallworld', 'wd_corr', 'total_abs_w', 'seconds')}, flush=True)
    return res


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if mode == 'test':
        run('comm', 0.5, 1, epochs=1, tag='_timing_test')
    elif mode == 'sweep':
        seeds = [1, 2, 3]
        for seed in seeds:
            run('none', 0.0, seed)                       # baseline
            run('l1', 0.01, seed)
            for s in (0.01, 0.1, 0.5, 1.0):               # strength sweep + ablation
                run('comm', s, seed)
                run('dist', s, seed)
        for seed in seeds:                                # distance power / delay length (strength 0.5)
            for dp in (0.5, 2):
                run('comm', 0.5, seed, distance_power=dp)
            for d in (20, 40):
                run('comm', 0.5, seed, delay=d)
    elif mode == 'next':
        # 1) more seeds (4-8) for the core conditions
        for seed in range(4, 9):
            run('none', 0.0, seed); run('l1', 0.01, seed)
            for s in (0.01, 0.1, 0.5):
                run('comm', s, seed)
        # 2) weaker strengths for distance-only (collapsed at >=0.01) and for comm
        for seed in range(1, 6):
            for s in (0.0001, 0.001):
                run('dist', s, seed)
        # 3) longer training (40 epochs) to see whether structure emerges later
        for seed in (1, 2, 3):
            for kind, s in (('none', 0.0), ('comm', 0.1), ('comm', 0.5), ('dist', 0.001)):
                run(kind, s, seed, epochs=40, tag=f'{kind}_s{s}_seed{seed}_ep40')
    elif mode == 'l1match':
        # paper's baseline is L1 matched on overall connectivity strength: sweep L1 so |W| covers the comm range
        for seed in range(1, 6):
            for s in (0.001, 0.003, 0.03, 0.05):
                run('l1', s, seed)
    elif mode == 'big':
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        core = (('none', 0.0), ('l1', 0.003), ('l1', 0.01), ('dist', 0.001), ('comm', 0.01), ('comm', 0.03), ('comm', 0.1), ('comm', 0.5))
        for seed in range(1, 25):                      # 10 epochs, seeds 1-24
            if seed % nparts != part: continue
            for kind, s in core: run(kind, s, seed)
        for seed in range(1, 11):                      # 40 epochs, seeds 1-10
            if seed % nparts != part: continue
            for kind, s in (('none', 0.0), ('l1', 0.01), ('comm', 0.03), ('comm', 0.1)):
                run(kind, s, seed, epochs=40, tag=f'{kind}_s{s}_seed{seed}_ep40')
