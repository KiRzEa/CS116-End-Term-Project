# Importing core libraries
import pandas as pd
from joblib import load, dump

import warnings
warnings.filterwarnings("ignore")

# Data processing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.mixture import GaussianMixture

# Save models function
def save_preprocessing_models(ordinal_encoder, pca, fa, gmm, save_path):
    models = {
        'ordinal_encoder': ordinal_encoder,
        'pca': pca,
        'fa': fa,
        'gmm': gmm
    }
    dump(models, save_path)

# Load models function
def load_preprocessing_models(load_path):
    models = load(load_path)
    return models['ordinal_encoder'], models['pca'], models['fa'], models['gmm']

def preprocess(X, split='train'):
    numerics = ['age', 'priors_count']
    categoricals = X.drop(numerics, axis=1).columns.tolist()

    if split == 'train':
        # Dealing with categorical data using OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        X[categoricals] = ordinal_encoder.fit_transform(X[categoricals])
        # Drop the original categorical columns

        fa = FactorAnalysis(n_components=len(numerics), rotation='varimax', random_state=0)
        fa.fit(X[numerics])

        extra_feats = [f'fa_{i}' for i in range(len(numerics))]
        X[extra_feats] = fa.transform(X[numerics])

        pca = PCA(n_components=len(numerics), random_state=0)
        pca.fit(X[numerics])

        pca_feats = [f'pca_{i}' for i in range(len(numerics))]

        X[pca_feats] = pca.transform(X[numerics])
        extra_feats += pca_feats
        # Fix the number of components for GMM
        num_gmm_components = 3  # You can adjust this based on your requirements

        dists = [num_gmm_components] * (len(numerics) + len(extra_feats))

        for feature, dist in zip(numerics + extra_feats, dists):
            x = X[feature].values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=dist, max_iter=300, random_state=0).fit(x)

            # Use GMM probabilities as features
            probs = gmm.predict_proba(x)
            prob_feats = [f'{feature}_gmm_prob_{i}' for i in range(dist)]
            X[prob_feats] = probs

        # Save models for later use
        save_preprocessing_models(ordinal_encoder, pca, fa, gmm, 'preprocessing_models.joblib')

    else:  # split == 'validation' or split == 'test'
        # Load models
        ordinal_encoder, pca, fa, gmm = load_preprocessing_models('preprocessing_models.joblib')

        # Dealing with categorical data using OrdinalEncoder
        X[categoricals] = ordinal_encoder.transform(X[categoricals])


        extra_feats = [f'fa_{i}' for i in range(len(numerics))]

        X[extra_feats] = fa.transform(X[numerics])
        pca_feats = [f'pca_{i}' for i in range(len(numerics))]

        X[pca_feats] = pca.transform(X[numerics])

        extra_feats += pca_feats
        # Fix the number of components for GMM
        num_gmm_components = 3  # Use the same number as in the training set

        dists = [num_gmm_components] * (len(numerics) + len(extra_feats))

        for feature, dist in zip(numerics + extra_feats, dists):
            x = X[feature].values.reshape(-1, 1)

            # Use GMM probabilities as features
            probs = gmm.predict_proba(x)
            prob_feats = [f'{feature}_gmm_prob_{i}' for i in range(dist)]
            X[prob_feats] = probs

    with open('selected_features.txt', 'r') as f:
        features = [line.strip() for line in f]
    return X[features]