import gluonts
# from gluonts.model.n_beats import NBEATSEstimator, NBEATSEnsembleEstimator
# from gluonts.trainer import Trainer
from gluonts.mx.trainer import Trainer
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from pathlib import Path
from gluonts.dataset.common import TrainDatasets, load_datasets
import mxnet as mx
import json
import pickle
from myNbeatsEnsemble import *
# name = "electricity_nips"
# name = "traffic_nips"
# dataset = get_dataset(name, regenerate=False)
# name = "electricity_nbeats_last7days"
# name = "traffic_nbeats_last7days"
# name = "electricity_nbeats_last7days_predlen1"
# name = "traffic_nbeats_last7days_predlen1"
# name = "tourism_monthly_predlen1"
# name = "tourism_monthly_predlen1_datefix"
name = "tourism_monthly_predlen24"
dataset_path = Path('./' + name)
dataset = load_datasets(
    metadata=dataset_path,
    train=dataset_path / "train",
    test=dataset_path / "test",
)

# NBEATS(I) settings
estimator = myNBEATSEnsembleEstimator(
    freq=dataset.metadata.freq,
    prediction_length=24,
    meta_context_length=[multiplier for multiplier in range(2, 8)],
    meta_loss_function=["MAPE"],
    meta_bagging_size=1,
    num_stacks=2,
    num_blocks=[3],
    num_block_layers=[4],
    widths=[256,2048],
    sharing=[True],
    expansion_coefficient_lengths=[2],
    stack_types=["T","S"],
    # l_h=20
    trainer=Trainer(# ctx=[mx.gpu(i) for i in range(mx.context.num_gpus())],
                    ctx="cpu",
                    # ctx=mx.gpu(2),
                    epochs=1,
                    learning_rate=1e-3,
                    batch_size=1024,
                    num_batches_per_epoch=300,)
                    # patience=1)
)
# predictor = estimator.train(training_data=dataset.train, shuffle_buffer_length=10240)
predictor = estimator.train(training_data=dataset.train)
predictor.serialize(Path("./nips_datasets/" + name + "/ensemble_batch300_predlen1"))

forecast_it, ts_it = make_evaluation_predictions(
dataset=dataset.test,  # test dataset
predictor=predictor,  # predictor
num_samples=1,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)
Yhats = [f_entry.samples[0] for f_entry in forecasts]
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))
print(json.dumps(agg_metrics, indent=4))

from gluonts.model.predictor import Predictor
predictor = Predictor.deserialize(Path("./nips_datasets/traffic_nbeats_last7days/ensemble_batch1000_predlen1"))

