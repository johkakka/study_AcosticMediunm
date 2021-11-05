import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GaussianMixture as gm


def get_meshgrid(x, y, nx, ny, margin=0.1):
    x_min, x_max = (1 + margin) * x.min() - margin * x.max(), (1 + margin) * x.max() - margin * x.min()
    y_min, y_max = (1 + margin) * y.min() - margin * y.max(), (1 + margin) * y.max() - margin * y.min()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    return xx, yy

def main():
    # 次元数を設定
    D = 2
    # クラスタ数を指定
    K = 2

    csv_input = pd.read_csv(filepath_or_buffer="data.csv", encoding="ms932", sep=",",
                            header=None, names=('x', 'y'))
    data = csv_input.values
    x = csv_input.iloc[:, 0]
    y = csv_input.iloc[:, 1]


    model = gm.GaussianMixtureModel(2)
    model.fit(data, max_iter=100, tol=1e-4, disp_message=True)
    labels = model.classify(data)

    colors = [(0.5, 0.5, 1), (1, 0.5, 0.5)]
    # plt.scatter(x, y, c=[colors[int(label)] for label in labels])
    # plt.scatter(model.Mu.T[0], model.Mu.T[1], c="k", marker="x")
    # plt.show()

    plt.plot()
    xx, yy = get_meshgrid(x, y, nx=500, ny=500, margin=0.1)
    Z = model.calc_prob_density(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.scatter(x, y, c=[colors[int(label)] for label in labels])
    plt.contour(xx, yy, Z)
    plt.scatter(model.Mu.T[0], model.Mu.T[1], c="k", marker="x")
    plt.show()

    print("Mu:\n", model.Mu)
    print("Sigma:\n",model.Sigma)



if __name__ == '__main__':
    main()