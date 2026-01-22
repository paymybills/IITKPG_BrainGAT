import glob
import numpy as np
from collections import Counter

files = sorted(glob.glob('abide_data/Outputs/cpac/nofilt_noglobal/rois_cc400/*.1D'))
print(f"Total files: {len(files)}\n")

shapes = []
for f in files[:100]:
    try:
        arr = np.loadtxt(f)
        shapes.append(arr.shape)
    except:
        shapes.append(None)

print("First 20 file dimensions:")
for i, (f, s) in enumerate(zip(files[:20], shapes[:20])):
    print(f"{f.split('/')[-1]}: {s}")

print(f"\nShape distribution (first 100 files):")
valid_shapes = [s for s in shapes if s is not None]
for shape, count in Counter(valid_shapes).most_common():
    print(f"  {shape}: {count} files")
