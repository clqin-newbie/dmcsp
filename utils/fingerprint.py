
# from matminer.featurizers.site import CrystalNNFingerprint  
import numpy as np 
import ase
from ase.spacegroup import Spacegroup, get_spacegroup
# fmt: off
import spglib
from itertools import combinations_with_replacement
from math import erf
from ase.geometry import get_angles
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Structure
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc


# from pymatgen.core.structure 
class OFPFingerprint:
    """Implementation of comparison using Oganov's fingerprint (OFP)
    functions, based on:

      * :doi:`Oganov, Valle, J. Chem. Phys. 130, 104504 (2009)
        <10.1063/1.3079326>`

      * :doi:`Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632
        <10.1016/j.cpc.2010.06.007>`

    Parameters:

    rcut: float
        Cutoff radius in Angstrom for the fingerprints.
        (Default 20 Angstrom)

    pbc: list of three booleans or None
         Specifies whether to apply periodic boundary conditions
         along each of the three unit cell vectors when calculating
         the fingerprint. The default (None) is to apply PBCs in all
         3 directions.

         Note: for isolated systems (pbc = [False, False, False]),
         the pair correlation function itself is always short-ranged
         (decays to zero beyond a certain radius), so unity is not
         subtracted for calculating the fingerprint. Also the
         volume normalization disappears.

    sigma: float
           Standard deviation of the gaussian smearing to be applied
           in the calculation of the fingerprints (in
           Angstrom). Default 0.02 Angstrom.

    nsigma: int
            Distance (as the number of standard deviations sigma) at
            which the gaussian smearing is cut off (i.e. no smearing
            beyond that distance). (Default 4)

    """

    def __init__(self, rcut=6, tcut=6, rbinnum=100,  tbinnum=100, sigma=0.02, nsigma=4, pbc=True):
        self.rcut = rcut
        self.rbinnum = rbinnum
        self.tbinnum = tbinnum
        self.pbc = pbc2pbc(pbc)
        self.sigma = sigma
        self.nsigma = nsigma
        self.tcut = tcut

    
    def get_rij(self, rcut, atoms, index):
        """获取指定index的临近列表"""
        pos = atoms.get_positions()
        nl = NeighborList([rcut]*len(atoms), skin=0.,
                    self_interaction=False, bothways=True)
        nl.update(atoms)
        indices, offsets = nl.get_neighbors(index)
        p = pos[indices] + np.dot(offsets, atoms.cell)
        r = cdist(p, [pos[index]])  # 计算Rij
        return r, p
    
    def _get_radial_fingerprints(self, atoms, index, add_r0=False, add_rel_size=False):
        distances = atoms.get_all_distances(mic=True)
        rmin = np.unique(distances)[1]
        rmax = np.unique(distances)[-1]
        # 获取r0
        r, _ = self.get_rij(5, atoms, index)
        r0 = r.min()
        #r标准化
        r, _ = self.get_rij(self.rcut*r0/2, atoms, index)
        r /= r0
        r = r-1
        binwidth = (self.rcut-1) / self.rbinnum
        m = int(np.ceil(self.nsigma * self.sigma / binwidth))
        x = 0.25 * np.sqrt(2) * binwidth * (2 * m + 1) * 1. / self.sigma
        smearing_norm = erf(x)
        bins = np.floor(r / binwidth) # 属于哪个bin
        rdf = np.zeros(self.rbinnum)
        for i in range(-m, m + 1):
            newbins = bins + i
            valid = np.where((newbins >= 0) & (newbins < self.rbinnum))  # 截断0~nbins
            valid_bins = newbins[valid].astype(int)
            values = np.array([1]*len(valid_bins), dtype=np.float64) 
            c = 0.25 * np.sqrt(2) * binwidth * 1. / self.sigma
            values *= 0.5 * erf(c * (2 * i + 1)) - \
                0.5 * erf(c * (2 * i - 1))
            values /= smearing_norm # 归一化
            # print(smearing_norm)
            for j, valid_bin in enumerate(valid_bins):
                rdf[valid_bin] += values[j]
        r0min = r0/rmin
        r0max = r0/rmax
        if add_r0:
            rdf = np.append(rdf, r0)
        if add_rel_size:
            rdf = np.append(rdf, r0min)
            rdf = np.append(rdf, r0max)
        # rdf /= len(indices)  # 原子平均
        return rdf
    
    def _get_angular_fingerprints(self, atoms, index, add_r0=False, add_rel_size=False):
        distances = atoms.get_all_distances(mic=True)
        rmin = np.unique(distances)[1]
        rmax = np.unique(distances)[-1]
        # 获取r0
        r, _ = self.get_rij(5, atoms, index)
        r0 = r.min()
        pos = atoms.get_positions()
        #r标准化
        r, p = self.get_rij(self.tcut*r0/2, atoms, index)
        idx_min = r.argmin()
        p_min = p[idx_min]
        p = np.delete(p, idx_min, axis=0)
        vecs1 = [pos[index] - p_min]   # 向量组1
        vecs2 = p-p_min        # 向量组2
        # 1. 计算余弦距离矩阵
        cos_dist = cdist(vecs1, vecs2, metric='cosine')
        # 2. 转成 弧度制夹角
        angles_rad = np.arccos(1 - cos_dist)
        binwidth = np.pi / self.tbinnum
        m = int(np.ceil(self.nsigma * self.sigma / binwidth))
        x = 0.25 * np.sqrt(2) * binwidth * (2 * m + 1) * 1. / self.sigma
        smearing_norm = erf(x)
        bins = np.floor(angles_rad / binwidth) # 属于哪个bin
        rdf = np.zeros(self.tbinnum)
        for i in range(-m, m + 1):
            newbins = bins + i
            valid = np.where((newbins >= 0) & (newbins < self.tbinnum))  # 截断0~nbins
            valid_bins = newbins[valid].astype(int)
            values = np.array([1]*len(valid_bins), dtype=np.float64) 
            c = 0.25 * np.sqrt(2) * binwidth * 1. / self.sigma
            values *= 0.5 * erf(c * (2 * i + 1)) - \
                0.5 * erf(c * (2 * i - 1))
            values /= smearing_norm # 归一化
            # print(smearing_norm)
            for j, valid_bin in enumerate(valid_bins):
                rdf[valid_bin] += values[j]
        r0min = r0/rmin
        r0max = r0/rmax
        if add_r0:
            rdf = np.append(rdf, r0)
        if add_rel_size:
            rdf = np.append(rdf, r0min)
            rdf = np.append(rdf, r0max)
        return rdf
    
    def get_unique_site_fingerprints(self, atoms, angular=True, radial=True, add_r0=False, add_rel_size=False):
        cell_spg = (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers()
        )
        dataset = spglib.get_symmetry_dataset(cell_spg, symprec=0.01, angle_tolerance=2.5)
        equivalent_atoms = dataset['equivalent_atoms']
        unique_atoms = np.unique(equivalent_atoms)
        atomic_num = atoms.get_atomic_numbers()
        sitesdic = {}
        Z_dict = {}
        for t in unique_atoms:
            sitesdic[t] = []
            for idx, i in enumerate(equivalent_atoms):
                if i == t:
                    sitesdic[t].append(idx)
                    Z_dict[t] = atomic_num[idx]
        a = atoms.copy()
        a.set_pbc(self.pbc)
      
        fingerprints = {}
        for t in unique_atoms:
            fingerprint = []

            for i in sitesdic[t]:
                if radial == True and angular == True:
                    fingerprint1 = self._get_radial_fingerprints(a, i)
                    fingerprint2 = self._get_angular_fingerprints(a, i, add_r0, add_rel_size)
                    fingerprint.append(np.hstack((fingerprint1, fingerprint2)))
                else:
                    if radial == True:
                        fingerprint.append(self._get_radial_fingerprints(a, i, add_r0, add_rel_size))
                    if angular == True:
                        fingerprint.append(self._get_angular_fingerprints(a, i, add_r0, add_rel_size))

            fingerprint = np.mean(fingerprint, axis=0) 
            fingerprints[t] = fingerprint
        return fingerprints, Z_dict

    def get_struc_fingerprints(self, atoms, angular=True, radial=True, add_r0=False, add_rel_size=False):
        fps = self.get_fingerprints_per_atom(atoms, angular=True, radial=True, add_r0=False, add_rel_size=False)
        fps = sum(fps.values()) / len(fps)
        return fps
    
    def get_fingerprints_per_atom(self, atoms, angular=True, radial=True, add_r0=False, add_rel_size=False):
        a = atoms.copy()
        a.set_pbc(self.pbc)
        fingerprints = {}
        for i in range(len(atoms)):
            if radial == True and angular == True:
                    fingerprint1 = self._get_radial_fingerprints(a, i)
                    fingerprint2 = self._get_angular_fingerprints(a, i, add_r0, add_rel_size)
                    fingerprint = np.hstack((fingerprint1, fingerprint2))
            else:
                if radial == True:
                    fingerprint = self._get_radial_fingerprints(a, i, add_r0, add_rel_size)
                if angular == True:
                    fingerprint = self._get_angular_fingerprints(a, i, add_r0, add_rel_size)
            fingerprints[i] = fingerprint
        return fingerprints
    

def get_site_fp(df, cnnf):
    """
    根据提供的结构获取独特位点的指纹及元素
    return [{'element': 56, "fp0": 0.1 ....}, {'element': 5, "fp0": 0.2 ....}]
    """
    id_struct, struct = df
    data_list = []
    data_dict = {}
    element_list = []

    # 获取每个独特site所有原子的指纹及元素序数
    for idx, site_value in enumerate(struct.get_symmetry_dataset()['orbits']):
        fp = cnnf.featurize(struct, idx)
        site_value = int(site_value)
        if data_dict.get(site_value):
            data_dict[site_value].append(fp)
        else:
            data_dict[site_value] = [fp]
            element_list.append(struct[idx].specie.Z)

   # 独特site的指纹根据原子求平均
    fp_list = []
    for k, v in data_dict.items():
        # print(np.mean(v, axis=0))
        fp = np.mean(v, axis=0).tolist()
        fp_list.append(fp)
    
    for element, fp in zip(element_list, fp_list):
        d_dict = {}
        d_dict['id'] = id_struct
        d_dict['element'] = element
        for i, v in enumerate(fp):
            d_dict[f'fp{i}'] = v
        data_list.append(d_dict)
    return data_list


def get_struct_fp(struct):
    """
    根据提供的结构获取结构的指纹
    return [fp]
    """
    fp_list = []
    # 获取每个独特site所有原子的指纹及元素序数
    for idx in range(len(struct)):
        fp = cnnf.featurize(struct, idx)
        fp_list.append(fp)
    return np.mean(fp_list, axis=0).tolist()