import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize(x, y):
    """
    draw live visualization for error vs epoch
    :param x: range of x axis
    :param y: list of error per epoch to be visualized
    :return: live graph for error per each epoch
    """
    plt.style.use('fivethirtyeight')

    def animate(i):
        x_values = pd.DataFrame(x)
        y_values = pd.DataFrame(y)
        plt.cla()
        plt.plot(x_values, y_values)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Error per epoch')
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    ani = FuncAnimation(plt.gcf(), animate, 5000)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.9)
    plt.close()


def draw(x, y):
    """
        visualize the given data
        :param x: range of x axis
        :param y: list of values to be visualized
        :return: graph for given data
        """
    plt.style.use('fivethirtyeight')

    def animate(i):
        x_values = pd.DataFrame(x)
        y_values = pd.DataFrame(y)
        plt.cla()
        plt.plot(x_values, y_values)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Error per epoch')
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

    ani = FuncAnimation(plt.gcf(), animate, 5000)

    plt.tight_layout()
    plt.show()
    plt.close()
