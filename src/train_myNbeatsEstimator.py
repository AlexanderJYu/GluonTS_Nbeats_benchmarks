import gluonts
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
from myNbeatsEstimator import *
from gluonts.model.predictor import Predictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='dataset name')
parser.add_argument('multiplier', type=int, help='multiplier for lookback')

# name = "electricity_nbeats_last7days"
# name = "traffic_nbeats_last7days"

def train_ensemble_estimator(multiplier, dataset):
    pred_len = 24
    estimator = myNBEATSEstimator(
        freq=dataset.metadata.freq,
        prediction_length=pred_len,
        context_length=pred_len*multiplier,
        loss_function="sMAPE",
        num_stacks=2,
        num_blocks=[3],
        num_block_layers=[4],
        widths=[256,2048],
        sharing=[True],
        expansion_coefficient_lengths=[3],
        stack_types=["T","S"],
        l_h=10,
        trainer=Trainer(
                        ctx="cpu",
                        epochs=1,
                        learning_rate=1e-3,
                        batch_size=1024,
                        num_batches_per_epoch=24000,
            )
        )
    predictor = estimator.train(training_data=dataset.train)
    return predictor

def serialize(predictor, multiplier, name):
    predictor.serialize(Path("./nips_datasets/" + name + "/multiplier_" + str(multiplier)))

def deserialize(multiplier, name):
    return Predictor.deserialize(Path("./nips_datasets/" + name + "/multiplier_" + str(multiplier)))

def evaluate(predictor, dataset):
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
    return Yhats

if __name__ == "__main__":

    args = parser.parse_args()
    multiplier = args.multiplier
    name = args.name
    print("name = " + str(name))
    print("multiplier = " + str(multiplier))

    dataset_path = Path('./' + name)
    dataset = load_datasets(
        metadata=dataset_path,
        train=dataset_path / "train",
        test=dataset_path / "test",
    )

    predictor = train_ensemble_estimator(multiplier, dataset)
    print("finished training")
    serialize(predictor, multiplier, name)
    print("finished serializing")
