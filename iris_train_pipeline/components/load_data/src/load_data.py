from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()

    np.save('iris.npy', iris)

if __name__ == '__main__':
    print('Loading data...')
    load_data()