import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV, SelectKBest

from propy import PyPro, CTD, PseudoAAC
import pfeature




def load_data(path, shuf=False):
    """Load protein sequence dataset."""

    df = pd.read_csv(path)
    df["input"] = df["input"].apply(lambda x: x.strip().replace(" ", ""))
    df = df[df["membrane"] != "U"] # remove uknown targets

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
            #"moreau-broto": desc.GetMoreauBrotoAuto(),
            #"moran": desc.GetMoranAuto(),
            #"broto": desc.GetGearyAuto(),
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
    split = int(len(features)*0.7)
    X_train = X[:split]; X_test = X[split:]
    y_train = y[:split]; y_test = y[split:]

    return X_train, y_train, X_test, y_test

def shuffle_data(X, y):
    print(X.shape, y.shape)
    shuf = np.random.RandomState(42).permutation(len(X))

    return X[shuf], y[shuf]


if __name__ == "__main__":

    sequences, y, classes = load_data("data/deeploc_per_protein_test.csv")

    print(len(sequences))

    #features, desc_names, feature_names = extract_descriptors(sequences)

    #np.save("descriptors_test.npy", [features, desc_names, feature_names], allow_pickle=True)

    # features, desc_names, feature_names = np.load("descriptors_train.npy", allow_pickle=True)


    # DATA

    sequences_train, y_train, classes = load_data("data/deeploc_per_protein_train.csv")
    X_train, _, _ = np.load("descriptors_test.npy", allow_pickle=True)
    sequences_test, y_test, classes = load_data("data/deeploc_per_protein_test.csv")
    X_test, _, _ = np.load("descriptors_test.npy", allow_pickle=True)

    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)

    # CLASSIFIERS

    #clf = RandomForestClassifier()
    clf = SVC(C=2)
    #clf = MLPClassifier()

    # DIM REDUCTION

    proj_model = PCA(n_components=1000)
    #proj_model = RFE(estimator=clf, n_features_to_select=1000, verbose=True)
    #proj_model = SelectKBest(k=1000)
    #clf = clf.fit(features, y)
    X_train = proj_model.fit_transform(X_train, y=y_train)
    X_test = proj_model.transform(X_test)


    clf.fit(X_train, y_train)


    pred = clf.predict(X_test)
    #pred = np.random.choice([0,1,2], size=(len(y),))
    ca = accuracy_score(y_test, pred)

    print(ca)






