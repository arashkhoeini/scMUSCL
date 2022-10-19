import os.path as osp
import os

def mkdir(p):
    if not osp.exists(p):
        os.makedirs(p)
        print('DIR {} created'.format(p))
    return p