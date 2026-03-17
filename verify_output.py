"""
verify_output.py — Validate the output contract for predictions_test.pt.gz.

Usage:
    python3 verify_output.py [--path results/predictions_test.pt.gz]

Exits with code 0 on PASS, 1 on any violation.
"""

import argparse
import gzip
import sys
import torch


REQUIRED_KEYS = {'name', 'signer', 'gloss', 'text', 'sign'}
SIGN_DIM = 183
MIN_LEN = 16
MAX_LEN = 300


def verify(path):
    print(f"Loading {path} ...")
    try:
        with gzip.open(path, 'rb') as f:
            predictions = torch.load(f, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"FAIL — could not load file: {e}")
        sys.exit(1)

    violations = []

    if not isinstance(predictions, list):
        print(f"FAIL — top-level object is {type(predictions)}, expected list")
        sys.exit(1)

    n = len(predictions)
    pred_lengths = []

    for i, item in enumerate(predictions):
        if not isinstance(item, dict):
            violations.append(f"[{i}] not a dict: {type(item)}")
            continue

        missing = REQUIRED_KEYS - set(item.keys())
        extra   = set(item.keys()) - REQUIRED_KEYS
        if missing:
            violations.append(f"[{i}] missing keys: {missing}")
        if extra:
            violations.append(f"[{i}] unexpected keys: {extra}")
        if missing:
            continue  # skip further checks if keys are wrong

        sign = item['sign']
        if not isinstance(sign, torch.Tensor):
            violations.append(f"[{i}] sign is {type(sign)}, expected torch.Tensor")
        elif sign.dtype != torch.float32:
            violations.append(f"[{i}] sign.dtype={sign.dtype}, expected torch.float32")
        elif sign.ndim != 2:
            violations.append(f"[{i}] sign.ndim={sign.ndim}, expected 2")
        elif sign.shape[1] != SIGN_DIM:
            violations.append(f"[{i}] sign.shape[1]={sign.shape[1]}, expected {SIGN_DIM}")
        else:
            pred_len = sign.shape[0]
            pred_lengths.append(pred_len)
            if not (MIN_LEN <= pred_len <= MAX_LEN):
                violations.append(f"[{i}] pred_length={pred_len} outside [{MIN_LEN},{MAX_LEN}]")

        gloss = item['gloss']
        if not isinstance(gloss, str):
            violations.append(f"[{i}] gloss is {type(gloss)}, expected str")
        elif gloss != gloss.upper():
            violations.append(f"[{i}] gloss is not all-uppercase: {repr(gloss[:60])}")

        text = item['text']
        if not isinstance(text, str):
            violations.append(f"[{i}] text is {type(text)}, expected str")
        elif text != text.lower():
            violations.append(f"[{i}] text is not all-lowercase: {repr(text[:60])}")

        if not isinstance(item['name'], str):
            violations.append(f"[{i}] name is {type(item['name'])}, expected str")
        if not isinstance(item['signer'], str):
            violations.append(f"[{i}] signer is {type(item['signer'])}, expected str")

    # Summary
    print(f"Total samples : {n}")
    if pred_lengths:
        print(f"pred_length   : min={min(pred_lengths)}, max={max(pred_lengths)}, "
              f"mean={sum(pred_lengths)/len(pred_lengths):.1f}")
    print(f"Violations    : {len(violations)}")

    if violations:
        print("\nViolation details:")
        for v in violations[:50]:
            print(f"  {v}")
        if len(violations) > 50:
            print(f"  ... ({len(violations) - 50} more)")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='results/predictions_test.pt.gz')
    args = parser.parse_args()
    verify(args.path)
