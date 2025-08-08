# utils/feature_generator.py

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from molfeat.trans.pretrained import PretrainedDGLTransformer

# GCADTI 관련 임포트. GCADTI 폴더 내 구조에 따라 경로 수정 필요.
# 원본 코드의 Model 클래스가 run_GCADTI.py에 있었다고 가정
from GCADTI.run_GCADTI import Model as GcaDtiModel
from GCADTI.DTIDataset import DTIDataset, graph_collate_func


class DtiPredictor:
    """GCADTI 모델을 감싸서 DTI 예측을 수행하는 클래스"""
    def __init__(self, model_path, prot_list_path, device):
        self.device = torch.device(device)
        self.model_wrapper = GcaDtiModel(modeldir='./dataloader/GCADTI', device=device)
        self.model_wrapper.load_pretrained(model_path, device)
        self.prot_list = pd.read_csv(prot_list_path, index_col=0)['sequence'].unique()

    def predict_for_smiles(self, smiles: str) -> torch.Tensor:
        """하나의 SMILES에 대해 DTI 벡터를 예측합니다."""
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
            
        pred_df = pd.DataFrame({
            'smiles': [smiles] * len(self.prot_list),
            'sequence': self.prot_list,
            'label': [0] * len(self.prot_list)
        })

        with torch.no_grad():
            pred_set = DTIDataset(pred_df)
            _, y_pred = self.model_wrapper.predict(pred_set)
        
        pred_df['label'] = y_pred
        df_sorted = pred_df.sort_values(by=['sequence'], ascending=[True])
        return torch.tensor(df_sorted['label'].values, dtype=torch.float32)

def generate_drug_features(smiles: str, dti_predictor: DtiPredictor) -> dict:
    """SMILES로부터 모든 약물 특징 벡터를 생성하여 딕셔너리로 반환합니다."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 1. Morgan Fingerprint
    morgan_vec = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    
    # 2. GIN Embedding
    gin_transformer = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)
    gin_vec = gin_transformer(mol)
    
    # 3. DTI Vector
    dti_vec = dti_predictor.predict_for_smiles(smiles)

    return {
        'morgan': torch.tensor(morgan_vec, dtype=torch.float32),
        'gin': torch.tensor(gin_vec[0], dtype=torch.float32),
        'dti': dti_vec
    }