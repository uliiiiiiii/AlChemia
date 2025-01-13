import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from chemml.datasets import load_organic_density
molecules, target, dragon_subset = load_organic_density()

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

xscale = StandardScaler()
yscale = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(dragon_subset.values, target.values, test_size=0.25, random_state=42)

X_train = xscale.fit_transform(X_train)
X_test = xscale.transform(X_test)
y_train = yscale.fit_transform(y_train)

from chemml.models import MLP

mlp = MLP(engine='tensorflow',nfeatures=X_train.shape[1], nneurons=[32,64,128], activations=['ReLU','ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=50, loss='mean_squared_error',
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='sgd')

mlp.fit(X = X_train, y = y_train)

mlp.save(path=".",filename="saved_MLP")

from chemml.utils import load_chemml_model
loaded_MLP = load_chemml_model("./saved_MLP_chemml_model.json")
print(loaded_MLP)

loaded_MLP.model

import numpy as np

y_pred = loaded_MLP.predict(X_test)
y_pred = yscale.inverse_transform(y_pred.reshape(-1,1))

from chemml.utils import regression_metrics

# For the regression_metrics function the inputs must have the same data type
metrics_df = regression_metrics(y_test, y_pred)
print("Metrics: \n")
print(metrics_df[["MAE", "RMSE", "MAPE", "r_squared"]])

from chemml.visualization import scatter2D, SavePlot, decorator
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
df["Actual"] = y_test.reshape(-1,)
df["Predicted"] = y_pred.reshape(-1,)
sc = scatter2D('r', marker='.')
fig = sc.plot(dfx=df, dfy=df, x="Actual", y="Predicted")

dec = decorator(title='Actual vs. Predicted',xlabel='Actual Density', ylabel='Predicted Density',
                xlim= (950,1550), ylim=(950,1550), grid=True,
                grid_color='g', grid_linestyle=':', grid_linewidth=0.5)
fig = dec.fit(fig)
# print(type(fig))
sa=SavePlot(filename='Parity',output_directory='images',
            kwargs={'facecolor':'w','dpi':330,'pad_inches':0.1, 'bbox_inches':'tight'})
sa.save(obj=fig)
fig.show()