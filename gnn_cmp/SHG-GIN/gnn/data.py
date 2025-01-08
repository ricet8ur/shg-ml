import csv
import json
from pymatgen.core.structure import Structure
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import os
import numpy as np
import torch

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}
    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]
class AtomCustomJSONInitializer(AtomInitializer):


    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {key: value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)
class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, step=0.2):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        atom_init_file = os.path.join(self.root_dir, 'cgcnn-embedding.json')
        assert os.path.exists(atom_init_file), 'cgcnn-embedding.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf1 = GaussianDistance(dmin=0, dmax=self.radius, step=step)
        self.gdf2 = GaussianDistance(dmin=-1, dmax=1, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target, gap_, gap0, gap1, gap2, gap3, gap4, gap5, gap6, gap7, gap8, gap9= \
        self.id_prop_data[idx]
        gap0, gap1, gap2, gap3, gap4 = int(gap0), int(gap1), int(gap2), int(gap3), int(gap4)
        gap5, gap6, gap7, gap8, gap9 = int(gap5), int(gap6), int(gap7), int(gap8), int(gap9)
        gap = np.hstack((gap0, gap1, gap2, gap3, gap4, gap5, gap6, gap7, gap8, gap9))
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(str(crystal[i].specie))
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)

        len_nbrs = np.array([len(nbr) for nbr in all_nbrs])

        indexes = np.where((len_nbrs < self.max_num_nbr))[0]

        for i in indexes:
            cut = self.radius
            curr_N = len(all_nbrs[i])
            while curr_N < self.max_num_nbr:
                cut += self.delta
                nbr = crystal.get_neighbors(crystal[i], cut)
                curr_N = len(nbr)
            all_nbrs[i] = nbr

        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx = torch.LongTensor(
            [
                list(map(lambda x: x[2], nbrs[: self.max_num_nbr]))
                for nbrs in all_nbrs
            ]
        )
        nbr_fea_d = torch.Tensor(
            [
                list(map(lambda x: x[1], nbrs[: self.max_num_nbr]))
                for nbrs in all_nbrs
            ]
        )

        cart_coords = torch.Tensor(np.array(
            [crystal[i].coords for i in range(len(crystal))]
        ))
        atom_nbr_fea = torch.Tensor(np.array(
            [
                list(map(lambda x: x[0].coords, nbrs[: self.max_num_nbr]))
                for nbrs in all_nbrs
            ]
        ))
        centre_coords = cart_coords.unsqueeze(1).expand(
            len(crystal), self.max_num_nbr, 3
        )
        dxyz = atom_nbr_fea - centre_coords
        r = nbr_fea_d.unsqueeze(2)
        angle_cosines = torch.matmul(
            dxyz, torch.swapaxes(dxyz, 1, 2)
        ) / torch.matmul(r, torch.swapaxes(r, 1, 2))
        nbr_fea_d = np.array(nbr_fea_d)
        nbr_fea_d = self.gdf1.expand(nbr_fea_d)
        nbr_fea_a = np.array(angle_cosines)
        nbr_fea_a = self.gdf2.expand(nbr_fea_a)
        bond_angles_sum = nbr_fea_a.sum(axis=1)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea_a = torch.Tensor(bond_angles_sum)
        nbr_fea_d = torch.Tensor(nbr_fea_d)
        nbr_fea_a = torch.Tensor(nbr_fea_a)
        nbr_fea = torch.cat((nbr_fea_d, nbr_fea_a), dim=2)
        nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        gap = torch.Tensor(gap)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), gap, target, cif_id

    def format_adj_matrix(self, adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)


class CIF_Lister(Dataset):
    def __init__(self, crystals_ids, full_dataset, df=None):
        self.crystals_ids = crystals_ids
        self.full_dataset = full_dataset

        self.material_ids = df.iloc[crystals_ids].values[:, 0].squeeze()

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self, original_dataset):
        names = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i = self.crystals_ids[idx]
        material = self.full_dataset[i]

        n_features = material[0][0]
        e_features = material[0][1]
        e_features = e_features.view(-1, 50)
        a_matrix = material[0][2]
        g_features = material[1]
        g_features = g_features.view(-1, 10)
        y = material[2]
        id = material[3]

        graph_crystal = Data(x=n_features, y=y, edge_index=a_matrix, edge_attr=e_features, g=g_features,
                             id=id)
        return graph_crystal