import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_input = pd.read_csv(filepath_or_buffer="data.csv", encoding="ms932", sep=",")
    x = csv_input.iloc[:, 0]
    y = csv_input.iloc[:, 1]

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y)

    ax.set_title('first scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.show()

if __name__ == '__main__':
    main()