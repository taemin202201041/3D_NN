"""Hand-designed connectivity structures for the seRNN task (original notebook code reused, not modified).

100 neurons on the 5x5x4 grid. Module c = all neurons whose first grid coordinate is c  -> 5 modules x 20 neurons (spatial slabs).
Structures (W = recurrent kernel, forced to W * mask; diagonal / self-connections always allowed):
  A   : every module fully connected inside; 2 "bridge" neurons per module (the 2 most central ones) are connected to the
        bridge neurons of all other modules. All inter-module traffic goes through the 10 bridge neurons.
  B1  : 1 hub neuron per module (most central). Hub <-> every neuron of its module (star); hubs fully connected to each other.
        Ordinary neurons have NO direct links to each other: every path goes through hubs.
  B2  : same as B1 with 2 hubs per module (hubs of all modules, including the 2 of the same module, fully connected).
  rA/rB1/rB2 : random symmetric mask with the SAME number of connections as A/B1/B2 (control: structure vs. just sparsity).
  none: ordinary dense RNN, no regulariser (control).
  comm : dense RNN + seRNN comm penalty 0.03 (reference for the anneal conditions).
  sA/sB1/sB2 : slow version: 30 epochs, 20 pruning steps of 5 % at epochs 2..21, 8 epochs fine-tuning.
  aA/aB1/aB2 : "grid -> structure": start dense with the seRNN penalty (comm, strength 0.03), after epoch 2 gradually delete the
        non-target connections (smallest |W| first; 20/40/60/80/100 % at epochs 2..6), then fine-tune to epoch 10.
Everything else (task, 10 epochs, Adam, batch 128, noise 0.05, 100 relu units) is identical to the original experiment.
"""
import json, os, sys, time
import numpy as np
from seRNN_sweep import *          # noqa: F401,F403  (tf, keras, mazeGeneratorI, SE1, SE1_sWc ... from the notebook copy)

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
ST_OUT = os.path.join(HERE, 'results', 'seRNN_struct')
os.makedirs(ST_OUT, exist_ok=True)
N = 100
GRID = np.stack(SE1(se1=0.01).coordinates, axis=1).astype('float32')
MOD = GRID[:, 0].astype(int)                                  # module id 0..4
GROUPS = [np.where(MOD == c)[0] for c in range(5)]


def special(g, k):
    """the k neurons of module g that are closest to the module centre"""
    d = np.linalg.norm(GRID[g] - GRID[g].mean(0), axis=1)
    return g[np.argsort(d, kind='stable')[:k]]


def build_mask(kind):
    M = np.zeros((N, N), bool)
    if kind == 'A':
        for g in GROUPS: M[np.ix_(g, g)] = True
        sp = np.concatenate([special(g, 2) for g in GROUPS])
        for i in sp:
            for j in sp:
                if MOD[i] != MOD[j]: M[i, j] = True
    elif kind in ('B1', 'B2'):
        k = 1 if kind == 'B1' else 2
        sp = np.concatenate([special(g, k) for g in GROUPS])
        for g in GROUPS:
            s = special(g, k); M[np.ix_(s, g)] = True; M[np.ix_(g, s)] = True
        M[np.ix_(sp, sp)] = True
    else:
        raise ValueError(kind)
    np.fill_diagonal(M, True)
    return M


def random_mask(like, rng):
    """symmetric random mask with the same number of off-diagonal connections (and full diagonal)"""
    iu = np.triu_indices(N, 1); n_pairs = int((np.triu(like, 1)).sum() + (np.triu(like.T, 1) & ~np.triu(like, 1)).sum())
    M = np.zeros((N, N), bool); sel = rng.choice(len(iu[0]), n_pairs, replace=False)
    M[iu[0][sel], iu[1][sel]] = True; M = M | M.T; np.fill_diagonal(M, True)
    return M


class MaskConstraint(keras.constraints.Constraint):
    def __init__(self, var): self.var = var
    def __call__(self, w): return w * tf.cast(self.var, w.dtype)


class Anneal(keras.callbacks.Callback):
    """after epochs 1..5 (0-based) delete 20/40/60/80/100 % of the non-target connections, smallest |W| first"""
    def __init__(self, mask_var, target, rnn, start=1, steps=5):
        super().__init__(); self.mv, self.target, self.rnn, self.start, self.steps = mask_var, target, rnn, start, steps
        self.nontarget_total = int((~target).sum())

    def on_epoch_end(self, epoch, logs=None):
        if not self.start <= epoch < self.start + self.steps: return
        frac = (epoch - self.start + 1) / self.steps
        keep_n = int(round(self.nontarget_total * (1 - frac)))
        W = np.abs(self.rnn.cell.recurrent_kernel.numpy()); mask = self.mv.numpy() > 0
        cand = np.where(~self.target & mask)
        order = np.argsort(-W[cand])                              # largest first -> keep these
        newmask = self.target.copy()
        keep = order[:keep_n]; newmask[cand[0][keep], cand[1][keep]] = True
        self.mv.assign(newmask.astype('float32'))
        self.rnn.cell.recurrent_kernel.assign(self.rnn.cell.recurrent_kernel * self.mv)


def run_struct(cond, seed, epochs=10, strength=0.03):
    name = f'st_{cond}_seed{seed}'
    path = os.path.join(ST_OUT, name + '.json')
    if os.path.exists(path): return json.load(open(path))
    np.random.seed(seed); tf.random.set_seed(seed + 1000); keras.backend.clear_session()
    gen = mazeGeneratorI(goal_presentation_steps=20, delay_steps=10, choices_presentation_steps=20)
    tr = gen.construct_tf_data(5120, 128); te = gen.construct_tf_data(2560, 128)
    ex = next(iter(tr))
    base = cond[1:] if cond[0] in 'ras' and cond != 'none' else cond
    if cond[0] == 's': epochs = 30
    mask_var = None; regu = None; target = None
    if cond == 'none': mask = None
    elif cond == 'comm': mask = None; regu = SE1_sWc(se1=strength)
    elif cond in ('A', 'B1', 'B2'): mask = build_mask(cond)
    elif cond[0] == 'r': mask = random_mask(build_mask(base), np.random.default_rng(seed + 777))
    elif cond[0] in 'as': target = build_mask(base); mask = np.ones((N, N), bool); regu = SE1_sWc(se1=strength)
    else: raise ValueError(cond)
    if mask is not None: mask_var = tf.Variable(mask.astype('float32'), trainable=False)
    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(stddev=0.05),
        tf.keras.layers.SimpleRNN(100, activation='relu', recurrent_initializer='orthogonal', return_sequences=False,
                                  recurrent_regularizer=regu, recurrent_constraint=MaskConstraint(mask_var) if mask_var is not None else None),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.build(input_shape=ex[0].shape)
    rnn = model.layers[1]
    if mask is not None and cond[0] not in 'as':                       # sparse start: masked orthogonal init, spectral radius rescaled to 1
        W0 = rnn.cell.recurrent_kernel.numpy() * mask
        rad = np.max(np.abs(np.linalg.eigvals(W0)))
        rnn.cell.recurrent_kernel.assign((W0 / max(rad, 1e-8)).astype('float32'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    cbs = [RNNWeightMatrixHistoryI(RNN_layer_number=1)]
    if cond[0] == 'a': cbs.append(Anneal(mask_var, target, rnn))
    if cond[0] == 's': cbs.append(Anneal(mask_var, target, rnn, start=2, steps=20))
    t0 = time.time()
    h = model.fit(tr, epochs=epochs, validation_data=te, callbacks=cbs, verbose=0)
    W = np.array(h.history['RNN_Weight_Matrix'][-1])
    off = ~np.eye(N, dtype=bool)
    res = dict(name=name, cond=cond, seed=seed, epochs=epochs, train_acc=float(h.history['accuracy'][-1]),
               test_acc=float(h.history['val_accuracy'][-1]), acc_curve=[float(x) for x in h.history['val_accuracy']],
               seconds=time.time() - t0, nonzero_offdiag=int((np.abs(W[off]) > 0).sum()),
               nonzero_total=int((np.abs(W) > 0).sum()),
               mask_edges=int(mask_var.numpy().sum()) if mask_var is not None else N * N)
    np.save(os.path.join(ST_OUT, name + '_W.npy'), W)
    json.dump(res, open(path, 'w'))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k in ('test_acc', 'seconds', 'nonzero_total')}, flush=True)
    return res


CONDS = ['none', 'comm', 'A', 'B1', 'B2', 'rA', 'rB1', 'rB2', 'aA', 'aB1', 'aB2', 'sA', 'sB1', 'sB2']

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'masks':
        for k in ('A', 'B1', 'B2'):
            M = build_mask(k); off = M & ~np.eye(N, dtype=bool)
            print(k, 'directed offdiag edges', int(off.sum()), 'density incl diag %.3f' % M.mean(), 'symmetric', bool((M == M.T).all()),
                  'in/out degree min/max', int(off.sum(1).min()), int(off.sum(1).max()), 'special', [list(special(g, 2 if k != 'B1' else 1)) for g in GROUPS][:2])
    elif mode == 'test':
        for c in sys.argv[2:]:
            r = run_struct(c, 9990, epochs=2)
            for f in os.listdir(ST_OUT):
                if 'seed9990' in f: os.remove(os.path.join(ST_OUT, f))
    elif mode == 'run':
        part, nparts, s0, s1 = [int(x) for x in sys.argv[2:6]]
        for seed in range(s0, s1 + 1):
            if seed % nparts != part: continue
            for c in CONDS: run_struct(c, seed)
