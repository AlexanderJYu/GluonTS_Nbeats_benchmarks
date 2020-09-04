import gluonts
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
import jax.numpy as np
from gluonts.dataset.util import to_pandas
import pickle
from pathlib import Path
from gluonts.dataset.common import TrainDatasets, load_datasets

# dataset = get_dataset("electricity", regenerate=True)
# name = "traffic_nips"
# name = "electricity_nips"a
# name = "solar_nips"
# name = "exchange_rate_nips"
# name = "wiki_rolling_nips"
# name = "electricity_nbeats_last7days"
# name = "traffic_nbeats_last7days"
# name = "electricity_nbeats_last7days_predlen1"
# name = "traffic_nbeats_last7days_predlen1"
# name = "tourism_monthly_predlen1"
name = "tourism_monthly_predlen24"
predlen=24

dataset_path = Path('./' + name)
dataset = load_datasets(
        metadata=dataset_path,
        train=dataset_path / "train",
        test=dataset_path / "test",
    )
# dataset = get_dataset("wiki-rolling_nips", regenerate=False)
# dataset = get_dataset(name, regenerate=False)

dataset_train_list = list(dataset.train)
dataset_test_list = list(dataset.test)
# rec_pred_len = dataset.metadata.prediction_length

dataset_train_np = [np.asarray(to_pandas(train_entry)) for train_entry in dataset_train_list]
dataset_test_np = [np.asarray(to_pandas(test_entry)) for test_entry in dataset_test_list]

X_test = [test_entry[-8*predlen:-1*predlen] for test_entry in dataset_test_np]
Y_test = [test_entry[-1*predlen:] for test_entry in dataset_test_np]

with open("./nips_datasets/" + name + "/" + name + "_X.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("./nips_datasets/" + name + "/" + name + "_Y.pkl", "wb") as g:
    pickle.dump(Y_test, g)

