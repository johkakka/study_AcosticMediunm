import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GaussianMixture as gm

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
    model.fit(data, max_iter=2, tol=1e-4, disp_message=True)
    labels = model.classify(data)

    colors = ["red", "blue", "green"]
    plt.scatter(x, y, c=[colors[int(label)] for label in labels])
    plt.show()


if __name__ == '__main__':
    main()