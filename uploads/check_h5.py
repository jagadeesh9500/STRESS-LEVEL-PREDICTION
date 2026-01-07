import h5py
import os

path = 'models/facialemotionmodel (1).h5'
print(f'File size: {os.path.getsize(path) / (1024*1024):.2f} MB')

try:
    with h5py.File(path, 'r') as f:
        print('Keys:', list(f.keys()))
except Exception as e:
    print(f'h5py error: {e}')
