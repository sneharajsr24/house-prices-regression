from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Safely fill missing categorical values with 'None'
        cat_fill_cols = ['PoolQC', 'Fence', 'FireplaceQu']
        for col in cat_fill_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                if 'None' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('None')
                df[col] = df[col].fillna('None')

        # Feature engineering
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        df['SeasonSold'] = df['MoSold'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        df['TotalBath'] = df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'] + df['FullBath'] + 0.5 * df['HalfBath']
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalOutsideSF'] = df['WoodDeckSF'] + df['TotalPorchSF']
        df['OverallGrade'] = df['OverallQual'] * df['OverallCond']

        # Safe encoding maps
        df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})
        df['PavedDrive'] = df['PavedDrive'].map({'Y': 2, 'P': 1, 'N': 0})
        df['Street'] = df['Street'].map({'Pave': 1, 'Grvl': 0})
        df['Alley'] = df['Alley'].map({'Pave': 2, 'Grvl': 1, 'NA': 0})

        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

        # Drop columns no longer needed
        drop_cols = [
            'YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF',
            'OverallQual', 'OverallCond'
        ]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Ensure remaining object columns are categorical
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        return df


class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='SalePrice', threshold=0.45):
        self.target_column = target_column
        self.threshold = threshold
        self.selected_features_ = []

    def fit(self, X, y=None):
        df = X.copy()
        if self.target_column not in df.columns and y is not None:
            df[self.target_column] = y

        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        corr = df_encoded.corr()[self.target_column]
        self.selected_features_ = corr[abs(corr) > self.threshold].drop(self.target_column).index.tolist()
        return self

    def transform(self, X):
        return X[self.selected_features_].copy()
