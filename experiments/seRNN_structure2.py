"""Round 2: structure experiments with the ORIGINAL seRNN loss (SE1_sWc, strength 0.03, grid distance) kept unchanged.
The loss is never modified; only the connectivity mask / pruning schedule is a factor.

Conditions (all: 100 neurons, same task, Adam, batch 128, noise 0.05):
  mA, mB1, mB2     : structure A/B1/B2 enforced from the start (mask) + original seRNN loss        (10 epochs)
  mrA, mrB1, mrB2  : same number of connections, random positions (control) + original seRNN loss (10 epochs)
  c30              : dense seRNN (comm) trained 30 epochs  (fair control for the 30-epoch pruning runs)
  fA, fB1, fB2     : start dense with seRNN loss; epochs 2-6 delete 20/40/60/80/100 % of the non-target connections
                     (smallest |W| first), then 23 epochs of fine-tuning                          (30 epochs)  -> fast cut, long recovery
  qA, qB1, qB2     : start dense with seRNN loss; epochs 2-21 delete connections on a cubic schedule
                     (cumulative fraction 1-(1-t)^3: big cuts early, tiny cuts at the end), 8 epochs fine-tuning (30 epochs)
Already available from round 1 (30 epochs, linear 5 %/epoch over epochs 2-21, 8 fine-tune epochs): sA, sB1, sB2.
"""
import json, os, sys, time
import numpy as np
import seRNN_structure as S
from seRNN_structure import tf, keras, N, build_mask, random_mask, MaskConstraint, mazeGeneratorI, SE1_sWc, RNNWeightMatrixHistoryI, HERE

OUT2 = os.path.join(HERE, 'results', 'seRNN_struct2')
os.makedirs(OUT2, exist_ok=True)


class Schedule(keras.callbacks.Callback):
    """after epoch e (0-based) set cumulative fraction of non-target edges removed to fracs[e]; smallest |W| removed first"""
    def __init__(self, mask_var, target, rnn, fracs):
        super().__init__(); self.mv, self.target, self.rnn, self.fracs = mask_var, target, rnn, fracs
        self.total = int((~target).sum())

    def on_epoch_end(self, epoch, logs=None):
        if epoch not in self.fracs: return
        keep_n = int(round(self.total * (1 - self.fracs[epoch])))
        W = np.abs(self.rnn.cell.recurrent_kernel.numpy()); mask = self.mv.numpy() > 0
        cand = np.where(~self.target & mask); order = np.argsort(-W[cand])
        new = self.target.copy(); keep = order[:keep_n]; new[cand[0][keep], cand[1][keep]] = True
        self.mv.assign(new.astype('float32')); self.rnn.cell.recurrent_kernel.assign(self.rnn.cell.recurrent_kernel * self.mv)


def fracs_for(kind):
    if kind == 'f': return {e: (e - 1) / 5 for e in range(2, 7)}                                   # 0.2 ... 1.0
    if kind == 'q': return {e: 1 - (1 - (e - 1) / 20) ** 3 for e in range(2, 22)}                 # cubic, ends at 1.0
    raise ValueError(kind)


def run2(cond, seed, strength=0.03):
    name = f'st2_{cond}_seed{seed}'; path = os.path.join(OUT2, name + '.json')
    if os.path.exists(path): return json.load(open(path))
    np.random.seed(seed); tf.random.set_seed(seed + 1000); keras.backend.clear_session()
    epochs = 10; kind = None; base = None
    if cond == 'c30': epochs = 30; mask = None
    elif cond[0] == 'm' and cond[1] != 'r': base = cond[1:]; mask = build_mask(base)
    elif cond[:2] == 'mr': base = cond[2:]; mask = random_mask(build_mask(base), np.random.default_rng(seed + 777))
    elif cond[0] in 'fq': kind = cond[0]; base = cond[1:]; epochs = 30; mask = np.ones((N, N), bool); target = build_mask(base)
    else: raise ValueError(cond)
    gen = mazeGeneratorI(goal_presentation_steps=20, delay_steps=10, choices_presentation_steps=20)
    tr = gen.construct_tf_data(5120, 128); te = gen.construct_tf_data(2560, 128); ex = next(iter(tr))
    mv = tf.Variable(mask.astype('float32'), trainable=False) if mask is not None else None
    model = tf.keras.models.Sequential([
        tf.keras.layers.GaussianNoise(stddev=0.05),
        tf.keras.layers.SimpleRNN(100, activation='relu', recurrent_initializer='orthogonal', return_sequences=False,
                                  recurrent_regularizer=SE1_sWc(se1=strength),                       # original seRNN loss, unchanged
                                  recurrent_constraint=MaskConstraint(mv) if mv is not None else None),
        tf.keras.layers.Dense(4, activation='softmax')])
    model.build(input_shape=ex[0].shape); rnn = model.layers[1]
    if mask is not None and kind is None:                  # sparse start: masked orthogonal init, spectral radius rescaled to 1
        W0 = rnn.cell.recurrent_kernel.numpy() * mask
        rnn.cell.recurrent_kernel.assign((W0 / max(np.max(np.abs(np.linalg.eigvals(W0))), 1e-8)).astype('float32'))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    cbs = [RNNWeightMatrixHistoryI(RNN_layer_number=1)]
    if kind: cbs.append(Schedule(mv, target, rnn, fracs_for(kind)))
    t0 = time.time(); h = model.fit(tr, epochs=epochs, validation_data=te, callbacks=cbs, verbose=0)
    W = np.array(h.history['RNN_Weight_Matrix'][-1])
    res = dict(name=name, cond=cond, seed=seed, epochs=epochs, train_acc=float(h.history['accuracy'][-1]), test_acc=float(h.history['val_accuracy'][-1]),
               acc_curve=[float(x) for x in h.history['val_accuracy']], seconds=time.time() - t0, nonzero_total=int((np.abs(W) > 0).sum()),
               mask_edges=int(mv.numpy().sum()) if mv is not None else N * N)
    np.save(os.path.join(OUT2, name + '_W.npy'), W); json.dump(res, open(path, 'w'))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k in ('test_acc', 'seconds', 'nonzero_total')}, flush=True)
    return res


CONDS2 = ['mA', 'mB1', 'mB2', 'mrA', 'mrB1', 'mrB2']   # reduced by the user's request; controls none / comm come from round 1 (same seeds, same pipeline)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        for c in sys.argv[2:]:
            r = run2(c, 9990); print(c, 'nonzero', r['nonzero_total'], 'edges', r['mask_edges'], 'curve', [round(x, 2) for x in r['acc_curve']][::3])
            for f in os.listdir(OUT2):
                if 'seed9990' in f: os.remove(os.path.join(OUT2, f))
    elif mode == 'run':
        part, nparts, s0, s1 = [int(x) for x in sys.argv[2:6]]
        for seed in range(s0, s1 + 1):
            if seed % nparts != part: continue
            for c in CONDS2: run2(c, seed)
