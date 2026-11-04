"""Export the independently computed NumPy screen without rerunning it."""
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
(p/'reference').mkdir(exist_ok=True)
for f in p.glob('*_H*.npz'):
    a=np.load(f)
    np.savetxt(p/'reference'/(f.stem+'.csv'),np.column_stack((a['q'].ravel(order='F'),a['reference'].ravel(order='F'))),delimiter=',',header='q,reference',comments='')
