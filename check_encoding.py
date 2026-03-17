"""
check_encoding.py — run once before data_process.py to confirm CSV encoding.
Usage: python3 check_encoding.py <path_to_csv>
"""
import sys
import argparse


def check(csv_path):
    umlaut_chars = {'ü', 'ä', 'ö', 'Ü', 'Ä', 'Ö', 'ß'}
    encodings = ['latin-1', 'utf-8-sig', 'iso-8859-1']

    for enc in encodings:
        try:
            with open(csv_path, encoding=enc) as f:
                lines = f.readlines()
            # find header
            header = lines[0].strip().split('|')
            try:
                trans_idx = header.index('translation')
            except ValueError:
                print(f"[{enc}] FAIL — no 'translation' column in header: {header}")
                continue

            samples = []
            for line in lines[1:6]:
                parts = line.strip().split('|')
                if len(parts) > trans_idx:
                    samples.append(parts[trans_idx])

            # check for correctly decoded umlauts
            joined = ' '.join(samples)
            has_umlauts = any(c in joined for c in umlaut_chars)
            # check for multi-byte artifacts (signs of wrong encoding)
            artifacts = any(ord(c) > 0x00FF and c not in umlaut_chars for c in joined)

            print(f"[{enc}] First 5 translations:")
            for s in samples:
                print(f"  {repr(s)}")
            if has_umlauts and not artifacts:
                print(f"[{enc}] PASS — umlauts decoded correctly\n")
            elif not has_umlauts:
                print(f"[{enc}] WARN — no umlauts found in sample (may still be correct)\n")
            else:
                print(f"[{enc}] FAIL — multi-byte artifacts detected\n")

        except Exception as e:
            print(f"[{enc}] ERROR — {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to PHOENIX14T pipe-delimited CSV')
    args = parser.parse_args()
    check(args.csv_path)
