# predictor.py (수정된 전체 코드)

import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader

# 각 모듈에서 필요한 클래스와 함수들을 임포트
from model.cdr_model import CDRModel
from dataset.omics_drug_response_dataset import OmicsDrugResponseInferenceDataset, collate_fn, OmicsDrugResponseDataset
from utils.feature_generator import generate_drug_features, DtiPredictor

class DrugResponsePredictor:
    """모든 모델과 데이터를 로드하고 예측 파이프라인을 실행하는 클래스"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("Initializing predictor... This may take a moment.")
        self.device = device
        
        self.cdr_model = CDRModel().to(self.device)
        cdr_weight_path = './weights/best_model.pt'
        self.cdr_model.load_state_dict(torch.load(cdr_weight_path, map_location=self.device, weights_only=True))
        self.cdr_model.eval()
        
        dti_weight_path = './GCADTI/model.pt'
        prot_list_path = './GCADTI/IC_test.csv'
        self.dti_predictor = DtiPredictor(dti_weight_path, prot_list_path, self.device)
        
        # --- ✨ 수정: joblib 로드 시 클래스를 현재 모듈에 등록 ---
        globals()['OmicsDrugResponseDataset'] = OmicsDrugResponseDataset
        self.cell_lines, self.omics_features = self._extract_unique_cell_omics('./data/omics_drug_dataset.joblib')
        
        # --- ✨ 수정: 올바른 참조 파일을 로드하고 인덱스 설정 ---
        # drug_all.csv는 약물 메타데이터용으로 따로 로드할 수 있습니다 (필요 시).
        # self.drug_meta = pd.read_csv('./data/drug_all.csv')
        self.ic50_ref = pd.read_csv('./data/20231023_092657_imputed_drugresponse.csv', index_col=0)
        
        self.cell_all = pd.read_csv('./data/model_list_20250630.csv')
        print("Predictor initialized successfully!")

    def _extract_unique_cell_omics(self, dataset_path: str):
        """Joblib 데이터셋에서 고유한 세포주 오믹스 데이터를 추출합니다."""
        dataset_full = joblib.load(dataset_path)
        cell_line_list = dataset_full.cell_lines
        omics_tensor_dict = dataset_full.omics_tensors
        
        omics_list = []
        for idx, cell_line in enumerate(cell_line_list):
            omics_features = {
                omics_type: omics_tensor_dict[omics_type][idx]
                for omics_type in omics_tensor_dict
            }
            omics_list.append(omics_features)
        
        return cell_line_list, omics_list

    def _analyze_drug_similarity(self, pred_df: pd.DataFrame):
        """ ✨ 수정: 올바른 데이터 형식에 맞게 유사도 분석 로직 전체 수정 """
        
        # pred_df 에는 ['cell_lines', 'pred_lnIC50'] 컬럼이 있음
        # self.ic50_ref 에는 (세포주 x 약물) 형태의 IC50 값이 들어있음
        
        # 비교를 위해 pred_df의 인덱스를 cell_lines로 설정
        pred_indexed = pred_df.set_index('cell_lines')
        
        # 공통 세포주 찾기
        common_cells = self.ic50_ref.index.intersection(pred_indexed.index)
        
        if len(common_cells) == 0:
            print("Warning: No common cell lines found for similarity analysis.")
            return pd.DataFrame()

        # 공통 세포주에 대해 데이터 정렬
        ic50_subset = self.ic50_ref.loc[common_cells]
        pred_subset = pred_indexed.loc[common_cells]
        
        y_pred = pred_subset['pred_lnIC50'].values
        
        scores = []
        # 이제 ic50_ref의 각 약물(열)에 대해 반복하며 점수 계산
        for drug_col in ic50_subset.columns:
            y_true = ic50_subset[drug_col].values

            # NaN 값이 있는 경우 비교에서 제외
            valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if np.sum(valid_indices) < 2: # 비교할 샘플이 2개 미만이면 건너뛰기
                continue

            y_true_valid = y_true[valid_indices]
            y_pred_valid = y_pred[valid_indices]

            # Normalized RMSE
            rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid)**2))
            norm_rmse = rmse / (y_true_valid.max() - y_true_valid.min()) if (y_true_valid.max() - y_true_valid.min()) > 0 else 1

            # Pearson CC (clip 0~1)
            pearson_corr = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]
            pearson_clip = np.clip(pearson_corr, 0, 1)

            # Score 계산
            score = 0.5 * (1 - min(norm_rmse, 1)) + 0.5 * pearson_clip

            scores.append({
                "drug": drug_col.split(';')[1] if ';' in drug_col else drug_col,
                "RMSE_norm": norm_rmse,
                "Pearson": pearson_clip,
                "Score": score
            })
        
        if not scores:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(scores).sort_values("Score", ascending=False).reset_index(drop=True)
        return result_df.head(15)

    def predict(self, smiles: str):
        """SMILES를 입력받아 최종 결과인 두 개의 DataFrame을 반환합니다."""
        if not smiles or not isinstance(smiles, str):
            raise ValueError("SMILES 문자열을 입력해주세요.")

        try:
            with torch.no_grad():
                drug_features = generate_drug_features(smiles, self.dti_predictor)
                
                dataset = OmicsDrugResponseInferenceDataset(self.cell_lines, self.omics_features, drug_features)
                loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=lambda b: collate_fn(b, device=self.device))
                
                all_preds = []
                for batch in loader:
                    output, _, _ = self.cdr_model(batch)
                    all_preds.append(output.cpu())
                
                merged_preds = torch.cat(all_preds).numpy()
                pred_df = pd.DataFrame({'cell_lines': self.cell_lines, 'pred_lnIC50': merged_preds})

                pred_df = pred_df.merge(
                    self.cell_all[['model_id', 'tissue', 'cancer_type']],
                    how='left', left_on='cell_lines', right_on='model_id'
                )
                pred_df = pred_df[['cell_lines', 'tissue', 'cancer_type', 'pred_lnIC50']]

                similarity_df = self._analyze_drug_similarity(pred_df)
                
                return pred_df, similarity_df
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            raise e