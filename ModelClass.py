import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union, List, Dict, Any

class PredictionModel:
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        model_filename_candidates: tuple = ("model.pkl", "model.joblib", "model.sav"),
        imputer_filename: str = "imputer.joblib",
        whisker_filename: str = "whisker_map.json",
        metadata_filename: str = "metadata.json",
        label_encoder_filename: str = "label_encoder.pkl",
    ):
        self.artifacts_dir = artifacts_dir
        self.model_filename_candidates = model_filename_candidates
        self.imputer_filename = imputer_filename
        self.whisker_filename = whisker_filename
        self.metadata_filename = metadata_filename
        self.label_encoder_filename = label_encoder_filename

        # loaded artifacts placeholders
        self.meta = None
        self.feature_map = None            # mapping canonical -> list of aliases
        self.rename_map = None             # inverted alias -> canonical
        self.numeric_cols = None
        self.imputer = None
        self.whisker_map = None
        self.model = None
        self.label_encoder = None

        # load everything on init
        self._load_artifacts()

    def _load_artifacts(self):
        meta_path = os.path.join(self.artifacts_dir, self.metadata_filename)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        self.meta = json.load(open(meta_path, "r"))

        # feature_map expected as canonical -> [aliases...]
        self.feature_map = self.meta.get("feature_map", {}) or {}
        # invert to alias -> canonical rename map for pandas.rename
        self.rename_map = {old: new for new, olds in self.feature_map.items() for old in olds}

        self.numeric_cols = self.meta.get("numeric_cols")
        if self.numeric_cols is None:
            raise ValueError("metadata.json must contain numeric_cols (list of numeric features).")

        # imputer
        imputer_path = os.path.join(self.artifacts_dir, self.imputer_filename)
        if not os.path.exists(imputer_path):
            raise FileNotFoundError(f"Missing imputer file: {imputer_path}")
        self.imputer = joblib.load(imputer_path)

        # whisker map
        whisker_path = os.path.join(self.artifacts_dir, self.whisker_filename)
        if not os.path.exists(whisker_path):
            raise FileNotFoundError(f"Missing whisker_map file: {whisker_path}")
        self.whisker_map = json.load(open(whisker_path, "r"))

        # model: try candidate filenames
        self.model = None
        for fname in self.model_filename_candidates:
            cand = os.path.join(self.artifacts_dir, fname)
            if os.path.exists(cand):
                try:
                    self.model = joblib.load(cand)
                except Exception:
                    with open(cand, "rb") as f:
                        self.model = pickle.load(f)
                break
        if self.model is None:
            raise FileNotFoundError(f"No model found in {self.artifacts_dir} matching {self.model_filename_candidates}")

        # label encoder (optional)
        le_path = os.path.join(self.artifacts_dir, self.label_encoder_filename)
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                self.label_encoder = pickle.load(f)

    def _preprocess_df(self, df: pd.DataFrame):
        """
        Apply column rename, dtype coercion, negative handling, imputation, and whisker clipping.
        Returns processed df.
        """
        df = df.copy()
        df_orig_index = df.index.tolist()

        # 1) Rename columns using inverted map (alias -> canonical)
        if self.rename_map:
            df = df.rename(columns=self.rename_map)

        # 2) Ensure numeric cols exist & coerce
        for col in self.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3) Handle negatives by rules
        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            if col in ["insol", "eqt", "teff"]:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    df.loc[neg_mask, col] = 0.0

            elif col in ["orbper", "trandep", "trandur", "rade", "rad"]:
                neg_mask = df[col] < 0
                if neg_mask.any():
                    df.loc[neg_mask, col] = np.nan

            elif col == "logg":
                pass  # keep as-is

        # 4) Impute using saved imputer and convert back to DataFrame safely
        imputed_arr = self.imputer.transform(df[self.numeric_cols])
        df[self.numeric_cols] = pd.DataFrame(imputed_arr, columns=self.numeric_cols, index=df.index)

        # 5) Clip outliers using whisker_map
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            val = self.whisker_map.get(col)
            if val is None:
                continue
            if isinstance(val, dict):
                upper = float(val.get("upper", np.inf))
                lower = float(val.get("lower", -np.inf))
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                upper = float(val[0])
                lower = float(val[1])
            else:
                continue

            high_mask = df[col] > upper
            low_mask = df[col] < lower
            if high_mask.any():
                df.loc[high_mask, col] = upper
            if low_mask.any():
                df.loc[low_mask, col] = lower

        return df

    def predict(
        self,
        data: Union[str, pd.DataFrame, dict, List[dict]],
        return_proba: bool = False,
        output_csv_path: Optional[str] = None
    ) -> Union[Dict[str, Any], str]:
        """
        Predict on the given data.
        - Single-row input -> returns dict (JSON-ready)
        - Multi-row / CSV -> saves CSV and returns path to CSV
        """
        # normalize to DataFrame
        if isinstance(data, str) and os.path.exists(data):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("data must be DataFrame, dict, list of dicts, or path to CSV")

        df_original = df.copy(deep=True)

        # preprocess
        df_proc = self._preprocess_df(df)

        # prepare feature matrix
        model_feature_order = self.meta.get("model_feature_order") or self.numeric_cols
        for col in model_feature_order:
            if col not in df_proc.columns:
                df_proc[col] = 0.0
        X = df_proc[model_feature_order].values

        # predict
        preds_idx = np.array(self.model.predict(X))

        # probabilities (if requested and supported)
        proba_list = None
        if return_proba and hasattr(self.model, "predict_proba"):
            proba_raw = self.model.predict_proba(X)
            if self.label_encoder is not None and hasattr(self.model, "classes_"):
                try:
                    class_labels = self.label_encoder.inverse_transform(self.model.classes_)
                except Exception:
                    class_labels = [str(c) for c in self.model.classes_]
            else:
                class_labels = [str(c) for c in getattr(self.model, "classes_", range(proba_raw.shape[1]))]
            proba_list = [
                {str(class_labels[c_i]): float(proba_raw[row_i, c_i])
                 for c_i in range(proba_raw.shape[1])}
                for row_i in range(proba_raw.shape[0])
            ]

        # decode labels if label encoder exists
        if self.label_encoder is not None:
            try:
                preds_decoded = self.label_encoder.inverse_transform(preds_idx)
            except Exception:
                preds_decoded = preds_idx
        else:
            preds_decoded = preds_idx

        n_rows = len(df_proc)
        # single-row -> return JSON/dict
        if n_rows == 1:
            out = {
                "prediction_index": int(preds_idx[0]) if np.issubdtype(preds_idx.dtype, np.integer) else preds_idx[0],
                "prediction": str(preds_decoded[0]),
                "probabilities": proba_list[0] if proba_list is not None else None
            }
            return out

        # multi-row -> append columns to original and save CSV
        df_out = df_original.reset_index(drop=True)
        df_out["prediction"] = preds_decoded
        # ensure integer dtype for indices when possible
        try:
            df_out["prediction_index"] = preds_idx.astype(int)
        except Exception:
            df_out["prediction_index"] = preds_idx

        if proba_list is not None:
            class_labels = list(proba_list[0].keys())
            for cl in class_labels:
                df_out[f"prob_{cl}"] = [p.get(cl, None) for p in proba_list]

        if not output_csv_path:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            output_csv_path = os.path.join(self.artifacts_dir, f"predictions_{timestamp}.csv")
        df_out.to_csv(output_csv_path, index=False)
        return output_csv_path
    
