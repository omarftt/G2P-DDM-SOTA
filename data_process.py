"""
data_process.py — Generate data files for G2P-DDM training with 61-keypoint skeletons.

Stage 1 (run before VQVAE training):
    python3 data_process.py --stage 1 \
        --pt_root dataset/phoenix14T \
        --csv_path <path_to_PHOENIX14T.csv> \
        --out_dir Data/ProgressiveTransformersSLP

Stage 2 (run after VQVAE training):
    python3 data_process.py --stage 2 \
        --pt_root dataset/phoenix14T \
        --csv_path <path_to_PHOENIX14T.csv> \
        --out_dir Data/ProgressiveTransformersSLP \
        --vqvae_ckpt <path_to_stage1_checkpoint.ckpt>

Output files per split (train/dev/test):
  {split}.skels  — one sample per line; each frame = 183 floats + 1 zero pad (184 total),
                   frames space-separated, all on one line.
  {split}.gloss  — one sample per line; space-separated uppercase gloss tokens.
  {split}.meta   — one JSON object per line; keys: name, signer, text.
  {split}.leng   — (Stage 2 only) one sample per line; space-separated int frame-counts
                   per gloss token.

Also writes (Stage 1):
  mean_183.npy, std_183.npy  — per-dim stats computed from training split only.
  src_vocab.txt              — gloss tokens only (NO special tokens — Dictionary adds
                               <blank>/<s>/<pad></s>/<unk>/<mask> automatically).
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path):
    """Return list of dicts: name, speaker, orth, translation."""
    rows = []
    with open(csv_path, encoding='latin-1') as f:
        header = f.readline().strip().split('|')
        idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < len(header):
                continue
            rows.append({
                'name':        parts[idx['name']],
                'speaker':     parts[idx['speaker']],
                'orth':        parts[idx['orth']],
                'translation': parts[idx['translation']],
            })
    return rows


def find_pt_file(pt_root, name):
    """Return (split_tag, path) or (None, None)."""
    for split in ('train_pt', 'dev_pt', 'test_pt'):
        path = os.path.join(pt_root, split, name + '.pt')
        if os.path.exists(path):
            return split.replace('_pt', ''), path
    return None, None


def load_sample(pt_path):
    """Returns (poses_flat (T,183), gloss_tokens, text, signer) or None."""
    sample = torch.load(pt_path, map_location='cpu', weights_only=False)
    poses = sample.get('poses_3d_filtered')
    if poses is None:
        return None
    T = poses.shape[0]
    if T < 16 or T > 300:
        return None

    gloss = sample.get('gloss', '')
    if isinstance(gloss, list):
        gloss_tokens = [g.upper() for g in gloss]
    else:
        gloss_tokens = [g.upper() for g in str(gloss).split()]

    text = str(sample.get('text', '')).lower()
    signer = str(sample.get('signer', ''))

    # (T, 61, 3) → (T, 183) C-contiguous: body[0:57], rhand[57:120], lhand[120:183]
    flat = poses.reshape(T, 183).numpy()
    return flat, gloss_tokens, text, signer


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(args):
    print("=== Stage 1: generating .skels, .gloss, .meta, src_vocab, stats ===")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_csv(args.csv_path)
    # splits[tag] = list of (name, flat(T,183), gloss_tokens, text, signer)
    splits = defaultdict(list)
    skipped = 0

    for row in rows:
        name = row['name']
        split, pt_path = find_pt_file(args.pt_root, name)
        if split is None:
            skipped += 1
            continue
        result = load_sample(pt_path)
        if result is None:
            skipped += 1
            continue
        flat, gloss_tokens, text, signer = result
        splits[split].append((name, flat, gloss_tokens, text, signer))

    for tag in ('train', 'dev', 'test'):
        print(f"  {tag}: {len(splits[tag])} samples")
    print(f"  skipped: {skipped}")

    # Normalization stats from training split only
    train_frames = np.concatenate([s[1] for s in splits['train']], axis=0)  # (N, 183)
    mean = train_frames.mean(axis=0)
    std  = train_frames.std(axis=0)
    np.save(os.path.join(args.out_dir, 'mean_183.npy'), mean)
    np.save(os.path.join(args.out_dir, 'std_183.npy'),  std)
    print(f"Saved mean/std over {len(train_frames)} training frames.")

    # Vocab — gloss tokens only. Do NOT include special tokens:
    # data/vocabulary.py Dictionary.__init__ pre-populates
    # <blank>, <s>, <pad>, </s>, <unk>, <mask> at indices 0-5.
    # add_from_file() appends each line directly; writing specials here
    # would corrupt their indices.
    vocab_tokens = set()
    for _, _, gloss_tokens, _, _ in splits['train']:
        vocab_tokens.update(gloss_tokens)
    vocab_path = os.path.join(args.out_dir, 'src_vocab.txt')
    with open(vocab_path, 'w') as f:
        for tok in sorted(vocab_tokens):
            f.write(tok + '\n')
    print(f"Wrote {len(vocab_tokens)} gloss tokens to {vocab_path}")

    # Write per-split files
    for split_tag, samples in splits.items():
        gloss_path = os.path.join(args.out_dir, f'{split_tag}.gloss')
        skels_path = os.path.join(args.out_dir, f'{split_tag}.skels')
        meta_path  = os.path.join(args.out_dir, f'{split_tag}.meta')

        with open(gloss_path, 'w') as fg, \
             open(skels_path, 'w') as fs, \
             open(meta_path,  'w') as fm:
            for name, flat, gloss_tokens, text, signer in samples:
                norm = (flat - mean) / (std + 1e-8)  # (T, 183)
                T = norm.shape[0]

                # .skels: each frame = 183 floats + 1 zero pad = 184 values
                frame_strs = []
                for t in range(T):
                    vals = ' '.join(f'{v:.6f}' for v in norm[t])
                    frame_strs.append(vals + ' 0')
                fs.write(' '.join(frame_strs) + '\n')

                # .gloss
                fg.write(' '.join(gloss_tokens) + '\n')

                # .meta
                fm.write(json.dumps({'name': name, 'signer': signer, 'text': text}) + '\n')

        print(f"[{split_tag}] wrote {len(samples)} samples.")

    print("Stage 1 complete.")


# ---------------------------------------------------------------------------
# Stage 2 — KMeans boundary detection (logic from modules/sequential_kmeans.py)
# ---------------------------------------------------------------------------

def _compute_lengths_kmeans(codes, n_gloss):
    """
    Given VQ codes (T, 3) and number of gloss tokens, return a list of
    T-summing frame-count integers per gloss.  Falls back to uniform split
    if KMeans peaks don't match n_gloss.
    """
    from sklearn.cluster import KMeans

    T = codes.shape[0]
    if n_gloss == 0:
        return []
    if n_gloss >= T:
        # each gloss gets 1 frame; pad last
        lengths = [1] * n_gloss
        lengths[-1] += T - n_gloss
        return lengths

    feats = codes.astype(float)
    kmeans = KMeans(n_clusters=n_gloss, random_state=0, n_init='auto').fit(feats)
    kmean_label = kmeans.labels_

    # Collect contiguous runs
    localdens = []
    pre_lab = kmean_label[0]
    cur_idx = [0]
    for lab_idx in range(1, T):
        if kmean_label[lab_idx] == pre_lab:
            cur_idx.append(lab_idx)
            if lab_idx == T - 1:
                localdens.append(cur_idx)
        else:
            localdens.append(cur_idx)
            pre_lab = kmean_label[lab_idx]
            cur_idx = [lab_idx]
            if lab_idx == T - 1:
                localdens.append(cur_idx)

    # Density peaks
    rho = {}
    for lnum in localdens:
        rho[math.floor(np.mean(lnum))] = len(lnum)
    sort_rho = sorted(rho.items(), key=lambda item: item[1], reverse=True)
    peaks = sorted([x[0] for x in sort_rho[:n_gloss]])

    def _uniform_split(T, n):
        base = T // n
        ls = [base] * n
        ls[-1] += T - sum(ls)
        return ls

    if len(peaks) != n_gloss:
        return _uniform_split(T, n_gloss)

    boundaries = [0]
    for p in range(len(peaks) - 1):
        pre, post = peaks[p], peaks[p + 1]
        middle = kmean_label[pre + 1: post + 1]
        for m in range(len(middle)):
            if middle[m] != kmean_label[pre]:
                boundaries.append(pre + 1 + m)
                break
    boundaries.append(T)
    lengths = [boundaries[i] - boundaries[i - 1] for i in range(1, len(boundaries))]

    if len(lengths) != n_gloss or sum(lengths) != T:
        return _uniform_split(T, n_gloss)

    return lengths


def run_stage2(args):
    print("=== Stage 2: generating .leng files using trained VQVAE ===")

    if not args.vqvae_ckpt:
        raise ValueError("--vqvae_ckpt is required for --stage 2")
    if not os.path.exists(args.vqvae_ckpt):
        raise FileNotFoundError(f"VQVAE checkpoint not found: {args.vqvae_ckpt}")

    mean = np.load(os.path.join(args.out_dir, 'mean_183.npy'))
    std  = np.load(os.path.join(args.out_dir, 'std_183.npy'))

    from stage1_models.pose_vqvae_sep import PoseVQVAE
    print(f"Loading VQVAE from {args.vqvae_ckpt} ...")
    vqvae = PoseVQVAE.load_from_checkpoint(args.vqvae_ckpt, strict=False)
    if not hasattr(vqvae, 'codebook'):
        vqvae.concat_codebook()
    vqvae.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae = vqvae.to(device)

    rows = load_csv(args.csv_path)
    splits = defaultdict(list)

    for row in rows:
        name = row['name']
        split, pt_path = find_pt_file(args.pt_root, name)
        if split is None:
            continue
        result = load_sample(pt_path)
        if result is None:
            continue
        flat, gloss_tokens, _, _ = result
        splits[split].append((name, flat, gloss_tokens))

    for split_tag, samples in splits.items():
        leng_path = os.path.join(args.out_dir, f'{split_tag}.leng')
        with open(leng_path, 'w') as fl:
            for name, flat, gloss_tokens in samples:
                norm = (flat - mean) / (std + 1e-8)
                T = norm.shape[0]
                skel_tensor = torch.FloatTensor(norm).unsqueeze(0).to(device)  # (1, T, 183)
                mask = torch.ones(1, T, dtype=torch.bool).to(device)

                with torch.no_grad():
                    vq_tokens, _, _, _ = vqvae.vqvae_encode(skel_tensor, mask)
                    codes = vq_tokens[0].cpu().numpy()  # (T, 3)

                lengths = _compute_lengths_kmeans(codes, len(gloss_tokens))
                fl.write(' '.join(str(l) for l in lengths) + '\n')

        print(f"[{split_tag}] wrote {len(samples)} samples to .leng")

    print("Stage 2 complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    parser.add_argument('--pt_root', type=str, default='dataset/phoenix14T',
                        help='Root directory containing train_pt/, dev_pt/, test_pt/')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to PHOENIX14T pipe-delimited CSV (latin-1 encoding)')
    parser.add_argument('--out_dir', type=str, default='Data/ProgressiveTransformersSLP')
    parser.add_argument('--vqvae_ckpt', type=str, default='',
                        help='[Stage 2 only] Path to trained Stage 1 VQVAE checkpoint')
    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(args)
    else:
        run_stage2(args)
