import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

class DenseTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = pd.Series(X.ravel()).fillna("").astype(str).tolist()
        self.vectorizer.fit(texts)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].fillna("").astype(str).tolist()
        else:
            texts = pd.Series(X.ravel()).fillna("").astype(str).tolist()
        return self.vectorizer.transform(texts).toarray()

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out()

class AutoMLDataProcessor:
    def __init__(self, target_column=None, task_type=None, date_col=None, forecast_horizon=1, nlp_config=None, scaler_type='standard', is_time_series=False, semi_supervised=False, strict_cv=False, enable_dfs=False, dfs_depth=1):
        self.target_column = target_column
        self.task_type = task_type
        self.date_col = date_col
        self.forecast_horizon = forecast_horizon
        self.nlp_config = nlp_config if nlp_config else {}
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.nlp_cols = []
        self.is_time_series = is_time_series or (task_type == 'time_series' or task_type == 'forecast')
        self.semi_supervised = semi_supervised
        self.strict_cv = strict_cv
        self.enable_dfs = enable_dfs
        self.dfs_depth = dfs_depth

    def _resolve_target_columns(self, df):
        if not self.target_column:
            return []
        if isinstance(self.target_column, list):
            return [c for c in self.target_column if c in df.columns]
        return [self.target_column] if self.target_column in df.columns else []

    def _apply_dfs(self, X):
        if self.enable_dfs and not self.is_time_series and not self.nlp_cols:
            try:
                import featuretools as ft
                logger.info("Applying Deep Feature Synthesis (DFS)...")
                dfs_df = X.copy()
                # Need an index for featuretools
                dfs_df['_dfs_id'] = range(len(dfs_df))
                es = ft.EntitySet(id="dataset")
                es = es.add_dataframe(dataframe_name="data", dataframe=dfs_df, index="_dfs_id")
                
                feature_matrix, _ = ft.dfs(
                    entityset=es,
                    target_dataframe_name="data",
                    max_depth=self.dfs_depth,
                    features_only=False,
                    verbose=False
                )
                if '_dfs_id' in feature_matrix.columns:
                    feature_matrix = feature_matrix.drop(columns=['_dfs_id'])
                
                logger.info(f"DFS completed. New feature matrix shape: {feature_matrix.shape}")
                return feature_matrix
            except ImportError:
                logger.warning("DFS failed: 'featuretools' is not installed. Please add it to requirements.")
            except Exception as e:
                logger.warning(f"DFS failed: {e}")
        return X

    def _apply_ts_features(self, df, y=None):
        df = df.copy()
        if self.date_col and self.date_col in df.columns:
            try:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
                df['hour'] = df[self.date_col].dt.hour
                df['dayofweek'] = df[self.date_col].dt.dayofweek
                df['quarter'] = df[self.date_col].dt.quarter
                df['month'] = df[self.date_col].dt.month
                df['year'] = df[self.date_col].dt.year
                df['dayofyear'] = df[self.date_col].dt.dayofyear
                df['dayofmonth'] = df[self.date_col].dt.day
                df['weekofyear'] = df[self.date_col].dt.isocalendar().week.astype(int)
            except Exception as e:
                logger.warning(f"Could not extract temporal features: {e}")

        target_vals = None
        if y is not None:
            target_vals = y
        else:
            targets = self._resolve_target_columns(df)
            if targets:
                target_vals = df[targets[0]]
            
        if target_vals is not None:
            target_vals_numeric = pd.to_numeric(target_vals, errors='coerce')
            if not target_vals_numeric.isna().all():
                target_vals = target_vals_numeric
                for i in range(self.forecast_horizon, self.forecast_horizon + 5):
                    df[f'lag_{i}'] = target_vals.shift(i)
                df[f'rolling_mean_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).mean()
                df[f'rolling_std_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).std()
                df = df.dropna()
        return df

    def fit_transform(self, df, nlp_cols=None):
        if df is None or df.empty:
            return df, None
        
        df = df.copy()
        nlp_cols = nlp_cols or []
        self.nlp_cols = [c for c in nlp_cols if c in df.columns]

        if self.is_time_series:
            df = self._apply_ts_features(df)

        target_cols = self._resolve_target_columns(df)
        if target_cols:
            X = df.drop(columns=target_cols)
            y = df[target_cols[0]] if len(target_cols) == 1 else df[target_cols].copy()
        else:
            X = df
            y = None

        X = self._apply_dfs(X)

        # Exclude date column from direct modeling if it still exists
        cols_to_fit = [c for c in X.columns if c != self.date_col]
        numeric_features = []
        categorical_low = []
        categorical_high = []

        for col in cols_to_fit:
            if col in self.nlp_cols:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                numeric_features.append(col)
            else:
                if X[col].nunique() < 15:
                    categorical_low.append(col)
                else:
                    categorical_high.append(col)

        transformers = []

        if self.strict_cv:
            # Bypass stateful scaling and imputation to prevent data leakage.
            # Rely on the underlying frameworks (AutoGluon, FLAML, TPOT) to handle it securely inside their CV loops.
            pass
        else:
            if numeric_features:
                num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', num_pipeline, numeric_features))

            if categorical_low:
                cat_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('cat_low', cat_pipeline, categorical_low))

            if categorical_high:
                cat_high_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
                transformers.append(('cat_high', cat_high_pipeline, categorical_high))

        # NLP vectorization using TF-IDF
        for text_col in self.nlp_cols:
            tfidf_max = self.nlp_config.get('max_features', 100)
            transformers.append((f'text_{text_col}', DenseTfidfVectorizer(max_features=tfidf_max), [text_col]))

        if transformers:
            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
            X_processed = self.preprocessor.fit_transform(X)
            # Reconstruct DataFrame with appropriate column names
            feature_names = []
            if numeric_features:
                feature_names.extend(numeric_features)
            if categorical_low:
                # get encoder out of column transformer
                ohe = self.preprocessor.named_transformers_['cat_low'].named_steps['onehot']
                feature_names.extend(list(ohe.get_feature_names_out(categorical_low)))
            if categorical_high:
                feature_names.extend(categorical_high)
            for text_col in self.nlp_cols:
                tfidf_transformer = self.preprocessor.named_transformers_[f'text_{text_col}']
                words = tfidf_transformer.get_feature_names_out([text_col])
                feature_names.extend([f"{text_col}_{w}" for w in words])
                
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        else:
            X_processed_df = X

        # y processing
        y_processed = None
        if y is not None:
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
                # Multi-task
                from sklearn.preprocessing import LabelEncoder
                y_processed_cols = {}
                for col in y.columns:
                    y_series = y[col]
                    if y_series.dtype == 'object' or y_series.dtype.name == 'category':
                        le = LabelEncoder()
                        y_processed_cols[col] = le.fit_transform(y_series.astype(str))
                    else:
                        y_processed_cols[col] = y_series.fillna(0).to_numpy()
                y_processed = pd.DataFrame(y_processed_cols, index=y.index)
            else:
                y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else pd.Series(y)
                unlabeled_mask = y_series.isna() | (y_series == -1) | (y_series == '-1') | (y_series == '')
                
                if self.task_type == 'classification' and self.semi_supervised:
                    from sklearn.preprocessing import LabelEncoder
                    labeled_y = y_series[~unlabeled_mask]
                    self.label_encoder = LabelEncoder()
                    if len(labeled_y) > 0:
                        encoded_labeled = self.label_encoder.fit_transform(labeled_y)
                    else:
                        encoded_labeled = []
                    y_processed = np.full(len(y_series), -1, dtype=int)
                    y_processed[~unlabeled_mask] = encoded_labeled
                    y_processed = pd.Series(y_processed, index=y_series.index)
                else:
                    if y_series.dtype == 'object' or y_series.dtype.name == 'category':
                        from sklearn.preprocessing import LabelEncoder
                        self.label_encoder = LabelEncoder()
                        y_processed = pd.Series(self.label_encoder.fit_transform(y_series), index=y_series.index)
                    else:
                        y_processed = y_series.fillna(0)
                        
        return X_processed_df, y_processed

    def transform(self, df):
        if df is None or df.empty:
            return df, None
            
        df = df.copy()
        if self.is_time_series:
            df = self._apply_ts_features(df)

        target_cols = self._resolve_target_columns(df)
        if target_cols:
            X = df.drop(columns=target_cols)
            y = df[target_cols[0]] if len(target_cols) == 1 else df[target_cols].copy()
        else:
            X = df
            y = None

        X = self._apply_dfs(X)

        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
            # Retrieve feature names from fit ColumnTransformer
            feature_names = []
            numeric_features = self.preprocessor.transformers_[0][2] if self.preprocessor.transformers_[0][0] == 'num' else []
            # Low category features
            cat_low_t = [t for t in self.preprocessor.transformers_ if t[0] == 'cat_low']
            categorical_low = cat_low_t[0][2] if cat_low_t else []
            # High category
            cat_high_t = [t for t in self.preprocessor.transformers_ if t[0] == 'cat_high']
            categorical_high = cat_high_t[0][2] if cat_high_t else []

            if numeric_features:
                feature_names.extend(numeric_features)
            if categorical_low:
                ohe = self.preprocessor.named_transformers_['cat_low'].named_steps['onehot']
                feature_names.extend(list(ohe.get_feature_names_out(categorical_low)))
            if categorical_high:
                feature_names.extend(categorical_high)
            for text_col in self.nlp_cols:
                tfidf_transformer = self.preprocessor.named_transformers_[f'text_{text_col}']
                words = tfidf_transformer.get_feature_names_out([text_col])
                feature_names.extend([f"{text_col}_{w}" for w in words])
                
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
        else:
            X_processed_df = X

        # y processing
        y_processed = None
        if y is not None:
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
                from sklearn.preprocessing import LabelEncoder
                y_processed_cols = {}
                for col in y.columns:
                    y_series = y[col]
                    if y_series.dtype == 'object' or y_series.dtype.name == 'category':
                        le = LabelEncoder()
                        y_processed_cols[col] = le.fit_transform(y_series.astype(str))
                    else:
                        y_processed_cols[col] = y_series.fillna(0).to_numpy()
                y_processed = pd.DataFrame(y_processed_cols, index=y.index)
            else:
                y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else pd.Series(y)
                unlabeled_mask = y_series.isna() | (y_series == -1) | (y_series == '-1') | (y_series == '')
                
                if self.task_type == 'classification' and self.semi_supervised:
                    if hasattr(self, 'label_encoder'):
                        labeled_mask = ~unlabeled_mask
                        y_processed = np.full(len(y_series), -1, dtype=int)
                        if labeled_mask.any():
                            # Transform only seen labels, set unseen/unlabeled to -1
                            labeled_y = y_series[labeled_mask]
                            y_processed[labeled_mask] = self.label_encoder.transform(labeled_y)
                        y_processed = pd.Series(y_processed, index=y_series.index)
                else:
                    if hasattr(self, 'label_encoder'):
                        y_processed = pd.Series(self.label_encoder.transform(y_series), index=y_series.index)
                    else:
                        y_processed = y_series.fillna(0)
        return X_processed_df, y_processed
