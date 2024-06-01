import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




def load_data(path, shuf=False):
    """Load protein sequence dataset."""

    df = pd.read_csv(path)
    df["input"] = df["input"].apply(lambda x: x.strip().replace(" ", ""))
    df = df[df["membrane"] != "U"] # remove seq with unknown targets
    df = df[~df["input"].str.contains("X")] # remove seq with unknown aminoacids
    df = df[~df["input"].str.contains("U")] # remove seq with unknown selenocystine
    df = df[~df["input"].str.contains("B")]

    if shuf:
        df = df.iloc[np.random.RandomState(42).permutation(len(df)), :]

    sequences = np.array(df["input"])
    classes, y = np.unique(df["membrane"], return_inverse=True)

    return sequences, y, classes


def extract_descriptors(sequences):
    """Extract all descriptors from a set of protein sequences.
    
    Args:
        sequences: A list of protein sequences as strings.

    Returns:
        features: A (n,d) array, where the i-th row are all concatenated descriptor features for i-th sequence.
        descriptor_names: A (d,) list of source descriptors for corresponding features.
                            E.g. "aminoacid_comp".
        features_names: A (d,) list of specific feature names for corresponding features.
                            E.g. "R" for Arganine count.
    """

    features = []
    descriptor_names = []
    feature_names = []


    for i,sequence in enumerate(sequences):
        desc = PyPro.GetProDes(sequence)

        if i%1 == 0:
            print(i)

        descriptors = {
            # aminoacid composition
            "aminoacid_comp": desc.GetAAComp(),
            "dipeptide_comp": desc.GetDPComp(),
            "tripeptide_comp": desc.GetTPComp(),
            # autocorrelation
            "moreau-broto": desc.GetMoreauBrotoAuto(),
            "moran": desc.GetMoranAuto(),
            "geary": desc.GetGearyAuto(),
            # CTD composition, transition, distribution
            "ctd_all": desc.GetCTD(),
            # conjoined triad
            #"conjoint_triad": pfeature.CalcConjointTriad(sequence),
            # sequence order
            ##"so_coupling_num": desc.GetSOCN(),
            ##"so_quasi": desc.GetQSO(),
            # pseudo aminoacid composition
            ##"type_i_paac": desc.GetPAAC(),
            ##"type_ii_paac": PseudoAAC.GetAPseudoAAC(sequence),
        }


        ft_vector = np.concatenate([list(descriptors[d].values()) for d in descriptors])
        features.append(ft_vector)
        
        if i == 0:
            for d in descriptors:
                descriptor_names += [d for i in range(len(descriptors[d]))]
                feature_names += list(descriptors[d].keys())
        
    features = np.stack(features, axis=0)

    return features, descriptor_names, feature_names


def train_test_split(X, y, ratio=0.7):
    split = int(len(X)*ratio)
    X_train = X[:split]; X_test = X[split:]
    y_train = y[:split]; y_test = y[split:]

    return X_train, y_train, X_test, y_test

def shuffle_data(X, y,random_state=42):
    print(X.shape, y.shape)
    shuf = np.random.RandomState(random_state).permutation(len(X))

    return X[shuf], y[shuf]


def perform_feature_selection(X, y):
    kf = KFold(n_splits=5)
    selected_features_sets = []

    for train_index, val_index in kf.split(X):
        X_train_split, X_val_split = X[train_index], X[val_index]
        y_train_split, y_val_split = y[train_index], y[val_index]

        # Feature Selection using Random Forest
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train_split, y_train_split)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_features = indices[:200]  # Select top 200 features
        selected_features_sets.append(set(selected_features))

    # Find intersection of selected features
    intersected_features = set.union(*selected_features_sets)
    intersected_features = list(intersected_features)
    print(f"Number of intersected features: {len(intersected_features)}")
    
    return intersected_features

def evaluate_model_with_pca(X, y, features, explained_variances):
    kf = KFold(n_splits=5)
    results = []

    for ev in explained_variances:
        fold_accuracies = []
        for train_index, val_index in kf.split(X):
            X_train_split, X_val_split = X[train_index], X[val_index]
            y_train_split, y_val_split = y[train_index], y[val_index]

            X_train_selected = X_train_split[:, features]
            X_val_selected = X_val_split[:, features]

            pca = PCA(n_components=ev, svd_solver='full')
            X_train_pca = pca.fit_transform(X_train_selected)
            X_val_pca = pca.transform(X_val_selected)

            clf = SVC(C=2)
            clf.fit(X_train_pca, y_train_split)
            pred = clf.predict(X_val_pca)
            ca = accuracy_score(y_val_split, pred)
            fold_accuracies.append(ca)
        
        avg_accuracy = np.mean(fold_accuracies)
        results.append((ev, avg_accuracy))
        print(f"Explained Variance: {ev}, Cross-Validation Accuracy: {avg_accuracy}")

    return results






if __name__ == "__main__":

    # sequences, y, classes = load_data("data/deeploc_per_protein_train.csv")

    # print(len(sequences))

    # features, desc_names, feature_names = extract_descriptors(sequences)

    # print(features.shape)

    # np.save("descriptors_train_ac.npy", [features, desc_names, feature_names], allow_pickle=True)

    # features, desc_names, feature_names = np.load("descriptors_train.npy", allow_pickle=True)


    # DATA

    sequences_train, y_train, classes = load_data("data/deeploc_per_protein_train.csv")
    X_train, desc_names, ft_names = np.load("descriptors_train_ac.npy", allow_pickle=True)
    sequences_test, y_test, classes = load_data("data/deeploc_per_protein_test.csv")
    X_test, _, _ = np.load("descriptors_test_ac.npy", allow_pickle=True)

    # Combine train and test data
    X_combined = np.concatenate((X_train, X_test), axis=0)
    y_combined = np.concatenate((y_train, y_test), axis=0)


    # IF YOU WANT TO USE THE SAME SPLIT AS THE PAPER COMMENT THIS PART
    # --------------------------------
    random_state = 5
    # Shuffle the combined dataset
    X_combined,y_combined = shuffle_data(X_combined, y_combined,random_state=random_state)
    from sklearn.model_selection import train_test_split
    # Split the combined dataset into new train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3,random_state=random_state)
    # --------------------------------
    # END COMMENT




    scaler = StandardScaler()
    fit_scaler = scaler.fit(X_train)
    X_train = fit_scaler.transform(X_train)
    X_test = fit_scaler.transform(X_test)

    # Perform feature selection using cross-validation
    intersected_features = perform_feature_selection(X_train, y_train)

    # Evaluate model with PCA using cross-validation
    explained_variances = [0.95, 0.8, 0.5, 0.2, 0.15]
    results = evaluate_model_with_pca(X_train, y_train, intersected_features, explained_variances)

    # Determine the best performing explained variance
    best_ev, best_accuracy = max(results, key=lambda item: item[1])
    print(f"Best Explained Variance: {best_ev}, Best Cross-Validation Accuracy: {best_accuracy}")

    # Apply PCA on the intersected features for the entire training set and test set
    X_train_selected = X_train[:, intersected_features]
    X_test_selected = X_test[:, intersected_features]

    pca = PCA(n_components=best_ev, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    print(f"Number of features after PCA: {X_train_pca.shape[1]}")

    # Hyperparameter tuning for SVM
    # param_grid = {
    #     'C': [0.1, 1,2,5, 10, 100],
    #     'gamma': [1, 0.1, 0.01, 0.001],
    #     'kernel': ['rbf', 'poly', 'sigmoid']


    

    
    #grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    #grid.fit(X_train_pca, y_train)
    
    # Best parameters from GridSearch
    #print(f"Best parameters: {grid.best_params_}")

    # Train the final classifier with best parameters on the entire training set
    #best_clf = grid.best_estimator_


    best_clf = SVC(C=2)
    best_clf.fit(X_train_pca, y_train)
    pred = best_clf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, pred)
    print(f"Test Accuracy: {test_accuracy}")

