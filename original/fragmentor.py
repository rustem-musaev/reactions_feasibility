# Transcribed from FRAGMENTOR_CODE.pdf (printed 27/11/2020).
# Kept as it was used; see reaction_feasibility/descriptors.py for the maintained version.
# Note: RDFRead was imported elsewhere in the original notebook/session.


def fit_fragmentor(filename):

    import pandas as pd
    from CIMtools.preprocessing import Fragmentor
    from sklearn.model_selection import train_test_split
    import pickle
    import numpy as np


    fragmentor = Fragmentor(fragment_type = 10, max_length = 5, min_length = 1,cgr_dynbonds=0)
    temp = []
    with RDFRead(filename, indexable = True) as f:
        for i in range(len(f)//1000+bool(len(f)%1000)):
            cgrs_list = []
            for x in f[i*1000:(i+1)*1000]:
                cgrs_list.append(~x)
            fragmentor.fit(cgrs_list)
        fragmentor.finalize()

    with open("final_fragmentor_wo_initials.pickle","wb") as p:
        pickle.dump(fragmentor, p)

    return fragmentor

def create_descriptors(filename, fragmentor):

    from compress import CompressShelve
    with RDFRead(filename, indexable = True) as f, CompressShelve("4_final_descriptors_wo_initials.shelve") as shelve:
        for k in range(len(f)//100+bool(len(f)%100)):
            cgrs_list = []
            class_values = []
            for y in f[k*100:(k+1)*100]:
                if 'Reconstructed' in y.meta['type']:
                    class_values.append(0)
                elif 'Decoy' in y.meta['type']:
                    class_values.append(1)
                else:
                    raise(print("something esle", y.meta))
                cgrs_list.append(~y)
            descriptor = fragmentor.transform(cgrs_list)
            if len(descriptor) == len(cgrs_list) and len(cgrs_list) == len(class_values):
                for n,class_value in enumerate(class_values):
                    index = n+(k*100)
                    desc = descriptor.iloc[[n]]
                    shelve[str(index)] = [desc, class_value]
