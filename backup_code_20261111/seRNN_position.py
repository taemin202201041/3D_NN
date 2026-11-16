"""seRNN with learnable neuron positions (original notebook code is reused, not modified).

Conditions: init in {grid, random} x positions in {fixed, learnable}.
Penalty = lambda * sum(|W| * D(P) * C(W)), same as SE1_sWc, but D is computed from the position tensor P.
To stop positions from collapsing to a single point (distance 0 => penalty vanishes), D is rescaled every step so that
its mean off-diagonal value equals the mean distance of the original 5x5x4 grid. Only the *arrangement* can change.
"""
import json, os, sys, time
import numpy as np
from seRNN_sweep import *          # noqa: F401,F403  (mazeGeneratorI, SE1, ... loaded from the notebook copy)

POS_OUT = os.path.join(HERE, 'results', 'seRNN_pos')
os.makedirs(POS_OUT, exist_ok=True)
N = 100
GRID = np.stack(SE1(se1=0.01).coordinates, axis=1).astype('float32')           # (100, 3) original grid
_gd = np.linalg.norm(GRID[:, None] - GRID[None], axis=-1)
GRID_D = _gd.astype('float32')
MEAN_GRID_D = float(_gd.sum() / (N * (N - 1)))
EYE = tf.eye(N)


def norm_dist(P):
    diff = P[:, None, :] - P[None, :, :]
    D = tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1) + 1e-9) * (1.0 - EYE)
    return D * (MEAN_GRID_D / (tf.reduce_sum(D) / (N * (N - 1))))


D_MIN = 0.7                                   # minimum allowed distance between two neurons (grid spacing is 1.0)
BOX_LO = np.zeros(3, 'float32'); BOX_HI = np.array([4, 4, 3], 'float32')   # same volume as the 5x5x4 grid


def raw_dist(P):
    diff = P[:, None, :] - P[None, :, :]
    return tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1) + 1e-9) * (1.0 - EYE)


def comm_matrix(A, comms_factor=1):
    s1 = tf.reduce_sum(A, axis=1)
    s3 = tf.linalg.diag(tf.pow(s1, -0.5))
    C = tf.linalg.set_diag(tf.linalg.expm(s3 @ A @ s3), tf.zeros(N))
    return C ** comms_factor


def spatial_metrics(W, D, seed=0):
    import networkx as nx
    A = np.abs(W); B = (A > np.quantile(A, 0.9)).astype(float); Bs = ((B + B.T) > 0).astype(float); np.fill_diagonal(Bs, 0)
    g = nx.from_numpy_array(Bs); comms = nx.community.louvain_communities(g, seed=seed)
    lab = np.zeros(N, int)
    for ci, c in enumerate(comms): lab[list(c)] = ci
    same = (lab[:, None] == lab[None]) & ~np.eye(N, dtype=bool)
    diff = lab[:, None] != lab[None]
    compact = float(D[same].mean() / D[diff].mean()) if same.any() and diff.any() else float('nan')
    iu = np.triu_indices(N, 1)
    return dict(wd_corr_final=float(np.corrcoef(A[iu], D[iu])[0, 1]), wd_corr_grid=float(np.corrcoef(A[iu], GRID_D[iu])[0, 1]),
                edge_len_final=float((B * D).sum() / max(B.sum(), 1)), edge_len_grid=float((B * GRID_D).sum() / max(B.sum(), 1)),
                compactness=compact, n_comms=len(comms), penalty_final=float((A * D).sum()))


def swap_opt(W, perm, rng, comms_factor=1, tries=3000):
    """Greedy slot swaps: neurons keep the 100 grid slots, only the neuron->slot assignment changes."""
    A = np.abs(W).astype('float64'); s1 = A.sum(1); s3 = np.diag(s1 ** -0.5)
    from scipy.linalg import expm
    C = expm(s3 @ A @ s3); np.fill_diagonal(C, 0); M = A * C ** comms_factor
    cost = float((M * GRID_D[np.ix_(perm, perm)]).sum()); n_acc = 0
    for _ in range(tries):
        i, k = rng.choice(N, 2, replace=False); q = perm.copy(); q[i], q[k] = q[k], q[i]
        c = float((M * GRID_D[np.ix_(q, q)]).sum())
        if c < cost: perm, cost, n_acc = q, c, n_acc + 1
    return perm, n_acc


def run_pos(init='grid', learn=True, strength=0.1, seed=1, epochs=10, lr_pos=0.02, comms_factor=1, swap=False, repel=0.0):
    """repel > 0 => continuous coordinates kept inside the grid volume with a minimum-distance repulsion (anti-collapse)."""
    name = f'pos_{init}_{"swap" if swap else ("cont" if (learn and repel > 0) else ("learn" if learn else "fixed"))}_s{strength}_seed{seed}'
    path = os.path.join(POS_OUT, name + '.json')
    if os.path.exists(path): return json.load(open(path))
    np.random.seed(seed); tf.random.set_seed(seed + 1000); keras.backend.clear_session()
    gen = mazeGeneratorI(goal_presentation_steps=20, delay_steps=10, choices_presentation_steps=20)
    tr = gen.construct_tf_data(5120, 128); te = gen.construct_tf_data(2560, 128)
    ex = next(iter(tr))
    rng = np.random.default_rng(seed + 500)
    P0 = GRID.copy() if init == 'grid' else rng.uniform([0, 0, 0], [4, 4, 3], (N, 3)).astype('float32')
    P = tf.Variable(P0, trainable=learn and not swap)
    perm = np.arange(N); swap_rng = np.random.default_rng(seed + 900)
    if swap: learn = False
    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(stddev=0.05),
        tf.keras.layers.SimpleRNN(100, activation='relu', recurrent_initializer='orthogonal', return_sequences=False),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.build(input_shape=ex[0].shape)
    rnn = model.layers[1]; Wvar = rnn.cell.recurrent_kernel
    opt_m = tf.keras.optimizers.Adam(); opt_p = tf.keras.optimizers.Adam(lr_pos)
    cce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            ce = cce(y, p)
            A = tf.abs(tf.convert_to_tensor(Wvar))
            pen = tf.reduce_sum(A * norm_dist(P) * comm_matrix(A, comms_factor))
            loss = ce + strength * pen
            if repel > 0:                                  # soft hard-sphere repulsion between every pair of neurons
                loss += repel * tf.reduce_sum(tf.nn.relu(D_MIN - raw_dist(P)) ** 2) / 2
        vars_m = model.trainable_variables
        grads = tape.gradient(loss, vars_m + ([P] if learn else []))
        opt_m.apply_gradients(zip(grads[:len(vars_m)], vars_m))
        if learn:
            opt_p.apply_gradients([(grads[-1], P)])
            if repel > 0: P.assign(tf.clip_by_value(P, BOX_LO, BOX_HI))   # stay inside the grid volume
        acc = tf.reduce_mean(tf.cast(tf.argmax(p, 1) == tf.argmax(y, 1), tf.float32))
        return ce, pen, acc

    def evaluate():
        c = n = 0
        for x, y in te:
            p = model(x, training=False); c += int(tf.reduce_sum(tf.cast(tf.argmax(p, 1) == tf.argmax(y, 1), tf.int32))); n += int(y.shape[0])
        return c / n

    t0 = time.time(); curve = []; trainacc = []
    for ep in range(epochs):
        accs = []
        for bi, (x, y) in enumerate(tr):
            accs.append(float(step(x, y)[2]))
            if swap and bi % 10 == 9:
                perm, _ = swap_opt(Wvar.numpy(), perm, swap_rng, comms_factor); P.assign(GRID[perm])
        trainacc.append(float(np.mean(accs))); curve.append(evaluate())
    W = Wvar.numpy(); Pf = P.numpy(); Df = norm_dist(tf.constant(Pf)).numpy()
    ref = structure_metrics(W, GRID_D, seed=seed)            # modularity / small-world / corr w.r.t. grid distance
    res = dict(name=name, init=init, learn=learn, swap=swap, strength=strength, seed=seed, epochs=epochs, lr_pos=lr_pos,
               train_acc=trainacc[-1], test_acc=curve[-1], acc_curve=curve, seconds=time.time() - t0,
               modularity=ref['modularity'], smallworld=ref['smallworld'], total_abs_w=ref['total_abs_w'],
               mean_displacement=float(np.linalg.norm(Pf - P0, axis=1).mean()))
    res.update(spatial_metrics(W, Df, seed=seed))
    np.save(os.path.join(POS_OUT, name + '_W.npy'), W); np.save(os.path.join(POS_OUT, name + '_P.npy'), Pf)
    json.dump(res, open(path, 'w'))
    print(name, {k: round(v, 3) for k, v in res.items() if k in ('test_acc', 'modularity', 'wd_corr_final', 'wd_corr_grid', 'compactness', 'mean_displacement', 'seconds')}, flush=True)
    return res


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        run_pos('grid', True, 0.1, 99, epochs=1)
        for f in os.listdir(POS_OUT):
            if 'seed99' in f: os.remove(os.path.join(POS_OUT, f))
    elif mode == 'conttest':
        run_pos('grid', True, 0.03, 98, epochs=2, repel=10.0, lr_pos=0.01)
        P = np.load(os.path.join(POS_OUT, 'pos_grid_cont_s0.03_seed98_P.npy')); D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        iu = np.triu_indices(N, 1); print('min pair dist %.3f mean %.3f (grid mean %.3f) box lo %s hi %s' % (D[iu].min(), D[iu].mean(), MEAN_GRID_D, P.min(0).round(2), P.max(0).round(2)))
    elif mode == 'cont':
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for seed in range(1, 41):
            if seed % nparts != part: continue
            run_pos('grid', True, 0.03, seed, repel=10.0, lr_pos=0.01)
    elif mode == 'swap':
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for seed in range(1, 11):
            if seed % nparts != part: continue
            for s in (0.03, 0.1): run_pos('grid', False, s, seed, swap=True)
    elif mode == 'sweep':
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for seed in range(1, 11):
            if seed % nparts != part: continue
            for s in (0.03, 0.1):
                for init in ('grid', 'random'):
                    for learn in (False, True):
                        run_pos(init, learn, s, seed)
    elif mode == 'more':
        # extra seeds (11-40) for the grid-fixed vs slot-swap accuracy comparison at strength 0.03
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for seed in range(11, 41):
            if seed % nparts != part: continue
            run_pos('grid', False, 0.03, seed)
            run_pos('grid', False, 0.03, seed, swap=True)
