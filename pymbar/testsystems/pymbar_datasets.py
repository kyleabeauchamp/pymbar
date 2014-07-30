import os
import numpy as np
import tables

def load_from_hdf(filename):
    f = tables.File(filename, 'r')
    u_kn = f.root.u_kn[:]
    N_k = f.root.N_k[:]
    f.close()
    return u_kn, N_k

def load_gas_data():
    root_dir = "/home/kyleb/src/choderalab/pymbar-datasets/"
    name = "gas-properties"
    u_kn, N_k = load_from_hdf(os.path.join(root_dir, name, "%s.h5" % name))
    return name, u_kn, N_k

def load_8proteins_data():
    root_dir = "/home/kyleb/src/choderalab/pymbar-datasets/"
    name = "8proteins"
    u_kn, N_k = load_from_hdf(os.path.join(root_dir, name, "%s.h5" % name))
    return name, u_kn, N_k
