import numpy as np
import shutil
from pydicom import dcmread
from tqdm import tqdm
from pathlib import Path


def swap_slices(pt, prefix='', folder_prefix='.'):

    new = folder_prefix / f'{prefix}_FLIPPED'
    if (new_pt := new.joinpath(pt.name)).exists():
        print(new_pt, "exists")
        return True
    new_pt.mkdir(exist_ok=True, parents=True)

    slope=[]
    intercept=[]

    for f in pt.iterdir():
        ds = dcmread(f)

        file_prefix = f.name.split('_')[0]

        slope.append(ds.RescaleSlope)
        intercept.append(ds.RescaleIntercept)

        gate = (ds.ImageIndex-1)//111
        my_index = 111*gate+(112-ds.ImageIndex+gate*111)
        opposite_image = '{}_{}.dcm'.format(file_prefix, "{0:04}".format(my_index))
        ds_other = dcmread(pt / opposite_image)
        ds.PixelData = ds_other.pixel_array.tobytes()
        ds.save_as(new_pt / f.name)
    if not len(np.unique(slope)) == 1 or not len(np.unique(intercept)) == 1:
        shutil.rmtree(new_pt)
        return False
    return True

wrong_folders = []
for folder in Path('/homes/michellef/Rb82_test_Sep2/test').iterdir():
    if not folder.name.endswith('dcm'):
        continue
    for pt in tqdm(list(folder.iterdir())):
        status = swap_slices(pt, prefix=folder.name, folder_prefix=folder.parent)
        if not status:
            wrong_folders.append(pt.name)
    print("Deleted converted folder - slope or intercept not the same in files for pts:")
    for p in wrong_folders:
        print(p)