import pickle

if __name__ == '__main__':
    outputs_mtmct_pkl = 'outputs/mtmct.pkl'
    with open(outputs_mtmct_pkl, 'rb') as file:
        mtmct = pickle.load(file)
    print(mtmct)