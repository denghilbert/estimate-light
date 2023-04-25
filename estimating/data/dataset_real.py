import os
import numpy as np
import imageio
import glob
from torch.utils.data import Dataset
import pdb, sys
import cv2

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class RealDataset(Dataset):
    def __init__(self, opt, dataroot):
        self.opt = opt
        self.dataroot = dataroot
        # _tmp_list = [os.path.split(path)[-1] for path in sorted(glob.glob(os.path.join(dataroot, '*_persp.*')))]
        _tmp_list = [os.path.split(path)[-1] for path in sorted(glob.glob(os.path.join(dataroot, '*.JPG')))]
        self.items = []
        for persp in _tmp_list:
            # if os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local.npy'))) and \
            #         os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local_0.hdr'))) and \
            #         os.path.exists(os.path.join(dataroot, persp.lower().replace('_persp.jpg', '_local_1.hdr'))):
            #     self.items.append(persp.split('_')[0])
            self.items.append(persp.split('.JPG')[0])

        print("dataset phase: real data test")
        print("num items: %d" %(len(self.items)))

    def get_item(self, idx):
        item = self.items[idx]
        img_idx = item
        persp_ldr_path = os.path.join(self.dataroot, '{}.JPG'.format(img_idx))

        persp = imageio.imread(persp_ldr_path, format='JPG') # H * W * C
        origin_size = persp.shape
        persp = cv2.resize(persp, (320, 240), interpolation=cv2.INTER_AREA)

        # for resize debugging
        # imageio.imwrite('/home/youmingdeng/inthewild/SOLD-Net/test.JPG', persp)
        # resize_back = cv2.resize(persp, (origin_size[1], origin_size[0]), interpolation=cv2.INTER_CUBIC)
        # imageio.imwrite('/home/youmingdeng/inthewild/SOLD-Net/test_back.JPG', resize_back)
        # ForkedPdb().set_trace()

        persp = np.transpose(np.asfarray(persp, dtype=np.float32), (2, 0, 1)) / 255.0 # C * H * W
        local_pos = np.array([-1, -1])
        local_panos = np.array([-1, -1])
        sun_pos = np.array([-1, -1])
        return_dict = {
            'color': persp,
            'meta': img_idx,
            'origin_size': origin_size
        }
        return return_dict

    # def get_item(self, idx):
    #     item = self.items[idx]
    #     img_idx = item
    #     persp_ldr_path = os.path.join(self.dataroot, '%s_persp.JPG' %(img_idx))
    #     persp = imageio.imread(persp_ldr_path, format='JPG') # H * W * C
    #     persp = np.transpose(np.asfarray(persp, dtype=np.float32), (2, 0, 1)) / 255.0 # C * H * W
    #     local_pos_path = os.path.join(self.dataroot, '%s_local.npy' %(img_idx))
    #     local_pos = np.load(local_pos_path)
    #     local_panos = []
    #     for local_id in range(local_pos.shape[0]):
    #         _local_pano_path = os.path.join(self.dataroot, '%s_local_%d.hdr' %(img_idx, local_id))
    #         _local_pano = imageio.imread(_local_pano_path, format='HDR-FI') # H * W * C
    #         _local_pano = np.transpose(np.asfarray(_local_pano, dtype=np.float32), (2, 0, 1)) # C * H * W
    #         local_panos.append(_local_pano)
    #     local_panos = np.stack(local_panos, axis=0)
    #     if os.path.exists(os.path.join(self.dataroot, '%s_sun_pos.txt' %(img_idx))):
    #         sun_pos = np.loadtxt(os.path.join(self.dataroot, '%s_sun_pos.txt' %(img_idx)))
    #     else:
    #         sun_pos = np.array([-1, -1])
    #     return_dict = {
    #         'color': persp,
    #         'local_pos': local_pos,
    #         'local_pano': local_panos,
    #         'sun_pos': sun_pos,
    #         'is_sunny': 1.0 if sun_pos[0] >= 0 else 0.0,
    #         'meta': img_idx
    #     }
    #     return return_dict


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
