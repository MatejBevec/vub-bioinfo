import numpy as np
import pandas as pd

from propy import PyPro, CTD, PseudoAAC
import pfeature




def load_data(path, shuf=True):
    """Load protein sequence dataset."""

    df = pd.read_csv(path)
    df["input"] = df["input"].apply(lambda x: x.strip().replace(" ", ""))

    if shuf:
        df = df.iloc[np.random.permutation(len(df)), :]

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





if __name__ == "__main__":

    sequences, y, classes = load_data("data/deeploc_per_protein_train.csv")

    features, desc_names, feature_names = extract_descriptors(sequences)

    np.save("descriptors_train.npy", [features, desc_names, feature_names], allow_pickle=True)

    features, desc_names, feature_names = np.load("descriptors_train.npy", allow_pickle=True)






