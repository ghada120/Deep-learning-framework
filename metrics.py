import numpy as np
import pandas as pd

currentDataClass = [1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2]
predictedClass =   [1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1]


def confusion_matrix(actual, predicted):
    classes = set(actual)
    number_of_classes = len(classes)

    conf_matrix = pd.DataFrame(np.zeros((number_of_classes, number_of_classes), dtype=int), index=classes, columns=classes)

    for i, j in zip(actual, predicted):
        conf_matrix.loc[i, j] += 1

    return conf_matrix


conf_matrix = confusion_matrix(currentDataClass, predictedClass)

print(conf_matrix)
print(conf_matrix.values)



