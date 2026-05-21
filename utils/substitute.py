import itertools
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
import torch
from matminer.featurizers.site import CrystalNNFingerprint  
from pandarallel import pandarallel
import numpy as np
from .fingerprint import get_site_fp
import pandas as pd
from pymatgen.core.periodic_table import Element
from .fingerprint import get_site_fp
from model import LinearModel
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path

def get_num_element_dict(composition):
    """
    获取不同原子数的元素
    input: 'UB2C2'
    return {1: ['U'], 2: ['B', 'C']}
    """
    data_dict = {}
    for k, v in composition.as_dict().items():
        v =  int(v)
        if data_dict.get(v):
            data_dict[v].append(k)
        else:
            data_dict[v] = [k]
    return data_dict

def interleave(a, b):
    """
    a = [i, j] 
    b = [1, 2]
    retrun [{i: 1, j: 2}, {i: 2, j: 1}]
    """
    data_list = []
    for p in itertools.permutations(b):
        res = {}
        for i, j in zip(a, p):
            res[i] = j
        data_list.append(res)
    return data_list

def get_templates_from_struc(struct, config, no_sub_elem):
    """
    从提供的结构获取不同组分的结构
    """
    atoms = get_atoms(struct)

    from utils.fingerprint import OFPFingerprint
    ofp = OFPFingerprint(rcut=6, tcut=6, rbinnum=100, tbinnum=100)  
    fps_dict, z_dict = ofp.get_unique_site_fingerprints(atoms)
    fp_list = list(fps_dict.values())
    z_list = list(z_dict.values())
    df_fp = pd.DataFrame(fp_list)
    df_fp['Z'] = pd.DataFrame(z_list)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel(config['input_dim'], config['latent_dim'], config['class_num']).to(device)
    model.load_state_dict(torch.load(f"./model/{config['model_name']}"))
    model.eval()
    fp_tensor = torch.tensor(df_fp.iloc[:, :-1].values, dtype=torch.float32).to(device)
    rs = torch.tensor(df_fp.iloc[:, -1:].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        prob_sites = model.get_site_prob(fp_tensor).to('cpu')
        # return fp_list, z_list
    
    prob_df = pd.DataFrame(prob_sites.tolist(), columns=range(1, prob_sites.shape[1]+1))
    
    # prob_df['id'] = df_fp['id']
    prob_df['element'] = df_fp['Z']
    # prob_df = pd.concat([prob_df, prob_df])
    def group_multiply(g):
        # g 是每个 id 对应的分组
        # 按每行的 element 取对应列的值

        values = g.iloc[:, :-1].prod()
        return values

    result = prob_df.groupby('element').apply(group_multiply)
    no_sub_elem = [Element(i).Z for i in no_sub_elem]
    index_list = []
    for r in result.index:
        if r not in no_sub_elem:
            index_list.append(list(result.columns))
        else:
            index_list.append([r])

    iter_list = list(itertools.product(*index_list))
    struc_list = []
    prob_list = []
    for c_list in iter_list:
        probs = 1
        subs_dict = {}
        for r, c in zip(result.index, c_list):
            probs *= result.loc[r, c]
            element = Element.from_Z(r)
            sub_element = Element.from_Z(c)
            subs_dict[element] = sub_element
            # for k in subs_dict.keys():
        new_struct = struct.copy().replace_species(subs_dict)
        struc_list.append(new_struct)
        prob_list.append(probs)
    result = pd.DataFrame([struc_list, prob_list]).T
    result.columns = ['structure', 'probability']
    result = result.sort_values(by='probability', ascending=False)
    result = result.reset_index(drop=True)
    result['id'] = result.index+1
    return result
    # result['id'] = result.index + 1
        # # merged = {k: v for d in i for k, v in d.items()}  # 合并字典
        # input()
        # 
        # print(probs)
        # input()
       
    
    


def get_templates_from_composition(target_formula, data):
    """
    从提供的组分获取不同模板的结构
    """
    # 组分筛选

    target_composition = Composition(target_formula)
    # 模板结构搜索
    template = data[data['anonymized_formula']==target_composition.anonymized_formula].reset_index()

    # 结构去重
    struc_list = []
    for idx in range(template.shape[0]):
        template_sturcture = Structure.from_str(template['structure'][idx], fmt='json')

    # sturc.replace_species({'Cl': 'B', "Zr": "U"})
        template_composition = template_sturcture.composition
        template_atom_num = template_composition.num_atoms
        target_atom_num = target_composition.num_atoms
        target_composition *= template_atom_num/target_atom_num

        target_num_element_dict = get_num_element_dict(target_composition)
        template_num_element_dict = get_num_element_dict(template_composition)

        # 获取每个元素的替代列表，相同原子的元素有多种组合
        substitute_list = []
        for k in target_num_element_dict.keys():
            target_element = target_num_element_dict[k]
            template_element = template_num_element_dict[k]
            tt_list = interleave(template_element, target_element)
            substitute_list.append(tt_list)

        # 替代列表的排列组合，生成替换结构
        for i in itertools.product(*substitute_list):
            target_structure = template_sturcture.copy()
            merged = {k: v for d in i for k, v in d.items()}  # 合并字典
            target_structure.replace_species(merged)
            struc_list.append(target_structure)
            
    return struc_list

            # 结构优化
def get_atoms(struc):
    atoms = AseAtomsAdaptor.get_atoms(struc)
    return atoms

def get_substitue_prob(struct_list, config):
    df_struct = pd.DataFrame({"structure": struct_list})
    df_struct['id'] = df_struct.index
    pandarallel.initialize(nb_workers=10)  # 用4核
    df_struct['atoms'] = df_struct['structure'].parallel_apply(get_atoms)

    from utils.fingerprint import OFPFingerprint
    ofp = OFPFingerprint(rcut=6, tcut=6, rbinnum=100, tbinnum=100)  
    def get_fps(atoms):
        fps_dict, z_dict = ofp.get_unique_site_fingerprints(atoms, add_rel_size=True)
        fp_list = list(fps_dict.values())
        z_list = list(z_dict.values())
        return fp_list, z_list
    
    rst = df_struct.iloc[:, :]['atoms'].parallel_apply(get_fps)

    fp_list = []
    z_list = []
    idx_list = []
    for idx, ij in enumerate(rst):
        fp_list += ij[0]
        z_list += ij[1]
        idx_list += [idx] * len(ij[1])

    # print(fp_list)
    # print(z_list)
    df_fp = pd.DataFrame(fp_list)
    df_fp['Z'] = pd.DataFrame(z_list)
    df_fp['id'] = pd.DataFrame(idx_list)
    # print(idx_list)
    # 结构概率评分
    probs_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModel(config['input_dim'], config['latent_dim'], config['class_num']).to(device)
    model.load_state_dict(torch.load(f"./model/{config['model_name']}"))
    model.eval()
    fp_tensor = torch.tensor(df_fp.iloc[:, :200].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        prob_sites = model.get_site_prob(fp_tensor).to('cpu')
    prob_df = pd.DataFrame(prob_sites.tolist(), columns=range(1, prob_sites.shape[1]+1))
    prob_df['id'] = df_fp['id']
    prob_df['element'] = df_fp['Z']
    
    def group_multiply(g):
        # g 是每个 id 对应的分组
        # 按每行的 element 取对应列的值
        values = [g.loc[idx, col] for idx, col in zip(g.index, g['element'])]
        # 乘积
        return np.mean(values)

    result = prob_df.groupby('id').parallel_apply(group_multiply).reset_index(name='probability')
    result['structure'] = df_struct['structure']
    result = result.sort_values(by='probability', ascending=False)
    result = result.reset_index(drop=True)
    result['id'] = result.index + 1
    return result

def write_results(results, out_path):
    outpath = Path(f"./output/{out_path}")
    outpath.mkdir(exist_ok=True)
    spg_info = []
    relax_spg_info = []
    formula_list = []
    for idx in range(results.shape[0]):
        struct = results['structure'][idx]
        struct.to(outpath.joinpath(f'{idx+1}_unrelax.vasp'), fmt='poscar')
        spg_syb, spg_num = struct.get_space_group_info()
        formula_list.append(struct.composition.formula)
        spg_info.append(spg_syb)
        if 'relax_structure' in results.columns:
            struct = results['relax_structure'][idx]
            struct.to(outpath.joinpath(f'{idx+1}_relax.vasp'), fmt='poscar')
            spg_syb, spg_num = struct.get_space_group_info()
            relax_spg_info.append(spg_syb)
        
    results['spg'] = spg_info
    results['formula'] = formula_list
    if 'relax_structure' in results.columns:
        results['spg_relax'] = relax_spg_info
        results[['id', "formula", 'spg', "spg_relax", 'probability']].to_csv(outpath.joinpath('info.csv'), index=False)
    else:
        results[['id', 'formula', 'spg', 'probability']].to_csv(outpath.joinpath('info.csv'), index=False)
