from randomForestClassifier import randomForest
from audioCharge import *

import numpy as np
import os

csv_path = os.path.join("csv", "train.csv")
csv2_path = os.path.join("csv", "train_soundscapes_labels.csv")
parent1 = "train_audio"
parent2 = "train_soundscapes"

#X1, Y1 , groups1 = build_dataset_principal(csv_path, parent=parent1, maxIter=50)
#X2, Y2 , groups2 = build_dataset(csv2_path, parent=parent2, maxIter=50)
#X1.extend(X2)
#Y1.extend(Y2)
#groups1.extend(groups2)


model, score, X_test, y_test = randomForest()

print("Score :", score)