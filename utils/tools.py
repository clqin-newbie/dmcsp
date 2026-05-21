import pandas as pd 
import numpy as np
import torch
import smact
from smact.screening import smact_filter
import matgl
from matgl.ext._ase_pyg import Relaxer


# 1. 定义相似度判断函数（可自定义）
def is_similar(fp1, fp2, tol=0):
    # 分数差值 <= tol 视为相似
    # print(s2)
    a = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
    simi = 0.5 * (1+a)
    return simi >= tol

# 2. 分组后组内去重（保留第一条）
def dedup_group(g):
    keep = []
    # 遍历组内每一行
    for i in range(len(g)):
        current = g.iloc[i]
        # 和已保留的行对比
        duplicate = False
        for k in keep:
            if is_similar(current['struct_fp'], k['struct_fp'], tol=0.99):
                duplicate = True
                break
        if not duplicate:
            keep.append(current)
    return pd.DataFrame(keep)

def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

def set_random_seed(seed): 
    '''Fixes random number generator seeds for reproducibility'''
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, batch_size, train_ratio):
    """划分数据集"""

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split( 
                                            dataset, [train_size, test_size]
                                            )
    
    print("train length:", train_size, "test length:",test_size)
    ##Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24)
    return train_loader, test_loader


def get_filted_composition(elements, threshold):
    
    # Convert element symbols into smact.Element objects
    element_objects = [smact.Element(e) for e in elements]

    # Apply smact_filter to find valid compositions within a stoichiometry threshold
    allowed_combinations = smact_filter(element_objects, threshold)
    formula_list = []
    for i in allowed_combinations:
        elem, _, num = i
        component = ''.join([elem + (str(n) if n != 1 else '') for elem, n in zip(elem, num)])
        formula_list.append(component)
    return formula_list

def relax_structure(struct):
    model = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
    relaxer = Relaxer(potential=model)

    relax_results = relaxer.relax(struct, fmax=0.1)
    # extract results
    final_structure = relax_results["final_structure"]
  
    final_energy = relax_results["trajectory"].energies[-1]
    
    return final_structure, final_energy