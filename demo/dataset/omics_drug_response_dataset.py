# dataset/dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset

# ==============================================================================
# 💡 해결책: .joblib 파일을 만들 때 사용한 원본 클래스를 여기에 정의합니다.
# 이 클래스가 있어야 joblib.load()가 객체를 성공적으로 복원할 수 있습니다.
# ==============================================================================
class OmicsDrugResponseDataset(Dataset):
    def __init__(self, weighted_ic50, crispr, pro, exp, meth, cid2morgan, cid2gin, cid2desc, cid2dti):
        """
        Args:
            weighted_ic50 (pd.DataFrame): (samples x drugs), column = drug CID
            crispr, pro, exp, meth (pd.DataFrame): (samples x features)
            cid2morgan (dict): {cid: morgan fingerprint numpy array}
            cid2gin (dict): {cid: gin embedding torch.Tensor or numpy array}
            cid2desc (dict): {cid: drug descriptor numpy array}
        """
        common_index = weighted_ic50.index \
                           .intersection(crispr.index) \
                           .intersection(pro.index) \
                           .intersection(exp.index) \
                           .intersection(meth.index)
        
        self.weighted_ic50 = weighted_ic50.loc[common_index]
        self.omics = {
            'crispr': crispr.loc[common_index],
            'pro': pro.loc[common_index],
            'exp': exp.loc[common_index],
            'meth': meth.loc[common_index]
        }

        self.cell_lines = list(self.weighted_ic50.index)
        self.drugs = list(self.weighted_ic50.columns)

        self.pairs = [
            (cell, drug)
            for cell in self.cell_lines
            for drug in self.drugs
        ]

        self.omics_tensors = {
            k: torch.tensor(v.values, dtype=torch.float32)
            for k, v in self.omics.items()
        }
        self.cell_line_to_idx = {cl: i for i, cl in enumerate(self.cell_lines)}
        
        self.drug_features = {
            cid: {
                'morgan': torch.tensor(cid2morgan[cid], dtype=torch.float32) if cid in cid2morgan else None,
                'gin': torch.tensor(cid2gin[cid], dtype=torch.float32) if cid in cid2gin else None,
                'desc': torch.tensor(cid2desc[cid], dtype=torch.float32) if cid in cid2desc else None,
                'dti': torch.tensor(cid2dti[cid], dtype=torch.float32) if cid in cid2dti else None,
            }
            for cid in self.drugs
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cell_line, drug_cid = self.pairs[idx]
        cell_idx = self.cell_line_to_idx[cell_line]
        omics_features = {k: v[cell_idx] for k, v in self.omics_tensors.items()}
        drug_feature = self.drug_features[drug_cid]
        target_ic50 = torch.tensor(
            self.weighted_ic50.loc[cell_line, drug_cid],
            dtype=torch.float32
        )
        return omics_features, drug_feature, cell_line, drug_cid, target_ic50


# ==============================================================================
# 아래는 예측(Inference) 단계에서 사용하는 간소화된 클래스입니다. (기존 코드)
# ==============================================================================
class OmicsDrugResponseInferenceDataset(Dataset):
    """
    Inference를 위한 Dataset.
    고정된 약물 특징과 다양한 세포주 오믹스 데이터를 결합합니다.
    """
    def __init__(self, cell_lines, omics_features_per_cell, drug_features):
        self.cell_lines = cell_lines
        self.omics_features_per_cell = omics_features_per_cell
        self.drug_features = drug_features

    def __len__(self):
        return len(self.cell_lines)

    def __getitem__(self, idx):
        omics_features = self.omics_features_per_cell[idx]
        cell_line_id = self.cell_lines[idx]
        return omics_features, self.drug_features, cell_line_id


def collate_fn(batch, device='cuda'):
    """
    배치 데이터를 모델 입력 형식에 맞게 딕셔너리로 변환합니다.
    """
    omics_list, drug_list, cell_lines = zip(*batch)
    
    batch_dict = {
        # Drug features
        'morgan': torch.stack([d['morgan'] for d in drug_list]).to(device),
        'gin':    torch.stack([d['gin'] for d in drug_list]).to(device),
        'dti':    torch.stack([d['dti'] for d in drug_list]).to(device),
        
        # Omics features
        'crispr': torch.stack([o['crispr'] for o in omics_list]).to(device),
        'pro':    torch.stack([o['pro'] for o in omics_list]).to(device),
        'exp':    torch.stack([o['exp'] for o in omics_list]).to(device),
        'meth':   torch.stack([o['meth'] for o in omics_list]).to(device),
    }
    return batch_dict