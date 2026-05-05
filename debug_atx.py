import re
from pathlib import Path


def extract_ascii(data, min_len=4, scan_limit=20000):
    strings, current = [], []
    for b in data[:scan_limit]:
        if 32 <= b <= 126:
            current.append(chr(b))
        else:
            if len(current) >= min_len:
                strings.append("".join(current))
            current = []
    if len(current) >= min_len:
        strings.append("".join(current))
    return strings


def extract_utf16(data, min_len=4, scan_limit=20000):
    strings, i = [], 0
    while i < min(len(data) - 1, scan_limit):
        if data[i + 1] == 0 and 32 <= data[i] <= 126:
            chars, j = [], i
            while j < len(data) - 1 and data[j + 1] == 0 and 32 <= data[j] <= 126:
                chars.append(chr(data[j]))
                j += 2
            if len(chars) >= min_len:
                strings.append("".join(chars))
            i = j
        else:
            i += 1
    return strings


p = Path("HI_Oahu_slr_final_dist.gdb/a00000001.TablesByName.atx")
data = p.read_bytes()
raw_ascii = extract_ascii(data, min_len=5)
raw_utf16 = extract_utf16(data, min_len=5)
print(f"ASCII strings: {len(raw_ascii)}, UTF16 strings: {len(raw_utf16)}")
for i, s in enumerate(raw_ascii[:5]):
    print(f"  ASCII[{i}] len={len(s)}: {repr(s[:120])}")
for i, s in enumerate(raw_utf16[:5]):
    print(f"  UTF16[{i}] len={len(s)}: {repr(s[:120])}")

# Test the regex split on the long strings
print("\n--- Testing re.split on long strings ---")
for s in raw_ascii:
    if len(s) > 50:
        parts = re.split(r'_{3,}', s)
        print(f"Split '{s[:60]}...' -> {len(parts)} parts: {[p[:20] for p in parts[:5]]}")
        break
