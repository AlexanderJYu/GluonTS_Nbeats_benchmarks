# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Here we reuse the datasets used by LSTNet as the processed url of the datasets
are available on GitHub.
"""
import json
import os
from pathlib import Path
from typing import List, NamedTuple, Optional
import pandas as pd
from gluonts.dataset.repository._util import metadata, save_to_file, to_dict
from gluonts.support.pandas import frequency_add
import numpy as np
from gluonts.support.util import get_download_path
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.common import TrainDatasets, load_datasets


default_dataset_path = get_download_path() / "datasets"


def load_from_pandas(
    df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    agg_freq: Optional[str] = None,
) -> List[pd.Series]:
    df = df.set_index(time_index)

    pivot_df = df.transpose()
    pivot_df.head()

    timeseries = []
    for row in pivot_df.iterrows():
        ts = pd.Series(row[1].values, index=time_index)
        if agg_freq is not None:
            ts = ts.resample(agg_freq).sum()
        first_valid = ts[ts.notnull()].index[0]
        last_valid = ts[ts.notnull()].index[-1]
        ts = ts[first_valid:last_valid]

        timeseries.append(ts)

    return timeseries


class LstnetDataset(NamedTuple):
    name: str
    url: str
    num_series: int
    num_time_steps: int
    prediction_length: int
    rolling_evaluations: int
    freq: str
    start_date: str
    agg_freq: Optional[str] = None


root = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/"

datasets_info = {
    "exchange_rate": LstnetDataset(
        name="exchange_rate",
        url=root + "exchange_rate/exchange_rate.txt.gz",
        num_series=8,
        num_time_steps=7588,
        prediction_length=30,
        rolling_evaluations=5,
        start_date="1990-01-01",
        freq="1B",
        agg_freq=None,
    ),
    "electricity": LstnetDataset(
        name="electricity",
        url=root + "electricity/electricity.txt.gz",
        # original dataset can be found at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        # the aggregated ones that is used from LSTNet filters out from the initial 370 series the one with no data
        # in 2011
        num_series=321,
        num_time_steps=26304,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2012-01-01",
        freq="1H",
        agg_freq=None,
    ),
    "electricity_nbeats_last7days": LstnetDataset(
        name="electricity_nbeats_last7days",
        url = None,
        num_series=370,
        num_time_steps=26304,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2012-01-01",
        freq="1H",
        agg_freq=None
    ),
    "electricity_nbeats_last7days_predlen1": LstnetDataset(
        name="electricity_nbeats_last7days_predlen1",
        url = None,
        num_series=370,
        num_time_steps=26304,
        prediction_length=1,
        rolling_evaluations=168,
        start_date="2012-01-01",
        freq="1H",
        agg_freq=None
    ),

    "traffic": LstnetDataset(
        name="traffic",
        url=root + "traffic/traffic.txt.gz",
        # note there are 963 in the original dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF
        # but only 862 in LSTNet
        num_series=862,
        num_time_steps=17544,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2015-01-01",
        freq="H",
        agg_freq=None,
    ),

    "traffic_nbeats_last7days": LstnetDataset(
        name="traffic_nbeats_last7days",
        url = None,
        num_series=963,
        num_time_steps=10560,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2015-01-01",
        freq="1H",
        agg_freq=None
    ),
    "traffic_nbeats_last7days_predlen1": LstnetDataset(
        name="traffic_nbeats_last7days_predlen1",
        url = None,
        num_series=963,
        num_time_steps=10560,
        prediction_length=1,
        rolling_evaluations=168,
        start_date="2015-01-01",
        freq="1H",
        agg_freq=None
    ),
    "tourism_monthly_predlen1": LstnetDataset(
        name="tourism",
        url = None,
        num_series=366,
        num_time_steps=None, # varying length time series
        prediction_length=1,
        rolling_evaluations=24,
        start_date="1992-01-01", # from row 2 of monthly_in.csv
        freq="M", # month end freq (could also be MS (month start freq))
        agg_freq=None
    ),
    "solar-energy": LstnetDataset(
        name="solar-energy",
        url=root + "solar-energy/solar_AL.txt.gz",
        num_series=137,
        num_time_steps=52560,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2006-01-01",
        freq="10min",
        agg_freq="1H",
    ),
}


def generate_lstnet_dataset(dataset_path: Path, dataset_name: str, numpyfile: str, csv_in=None, csv_out=None):
    ds_info = datasets_info[dataset_name]

    os.makedirs(dataset_path / dataset_name, exist_ok=True)

    with open(dataset_path / dataset_name / "metadata.json", "w") as f:
        f.write(
            json.dumps(
                metadata(
                    cardinality=ds_info.num_series,
                    freq=ds_info.freq,
                    prediction_length=ds_info.prediction_length,
                )
            )
        )

    train_file = dataset_path / dataset_name / "train" / "data.json"
    test_file = dataset_path / dataset_name / "test" / "data.json"

    
    if ds_info.url != None:
        df = pd.read_csv(ds_info.url, header=None)
    else:
        # df_npy = np.load('/Users/alexjyu/Desktop/electricity.npy')
        # df_npy = np.load('./electricity.npy')
        if numpyfile != None:
            df_npy = np.load(numpyfile)
            df = pd.DataFrame(data=df_npy)
            assert df.shape == (
                ds_info.num_time_steps,
                ds_info.num_series,
            ), f"expected num_time_steps/num_series {(ds_info.num_time_steps, ds_info.num_series)} but got {df.shape}"
        else:
            df_in = pd.read_csv('./tourism_alt/monthly_in.csv')
            df_out = pd.read_csv('./tourism_alt/monthly_oos.csv')

    if numpyfile != None:
        time_index = pd.date_range(
            start=ds_info.start_date,
            freq=ds_info.freq,
            periods=ds_info.num_time_steps,
        )

        timeseries = load_from_pandas(
            df=df, time_index=time_index, agg_freq=ds_info.agg_freq
        )

    if numpyfile == None and csv_in != None and csv_out != None:
        train_ts = []
        test_ts = []
        for i in range(1, ds_info.num_series+1):
        # for i in range(1,2):
            time_index = pd.date_range(start=df_in['m' + str(i)][1],
                                       freq=ds_info.freq,
                                       periods=df_in.count()['m' + str(i)] - 3)
            time_index_out = pd.date_range(start=frequency_add(time_index[-1], ds_info.prediction_length),
                                           freq=ds_info.freq,
                                           periods=24)
            # print("time_index = " + str(time_index))
            print("time_index[0] = " + str(time_index[0]))
            print("time_index[-1] = " + str(time_index[-1]))
            print("time_index_out = " + str(time_index_out))
            '''
            timeseries = load_from_pandas(df=df_in['m' + str(i)], 
                                          time_index=time_index, 
                                          agg_freq=ds_info.agg_freq)'''
            timeseries = df_in['m' + str(i)][3:]
            print("cat = " + str(i))
            print("total len(timeseries) = " + str(len(timeseries)))
            first_valid = timeseries[timeseries.notnull()].index[0]
            last_valid = timeseries[timeseries.notnull()].index[-1]
            print("timeseries.iloc[0:3] = " + str(timeseries.iloc[0:3]))
            print("first_valid = " + str(first_valid))
            print(timeseries[first_valid])
            print("last_valid = " + str(last_valid))
            print(timeseries[last_valid])
            # print(timeseries[last_valid+1])
            print("timeseries[0:3] BEFORE = " + str(timeseries[0:3]))
            timeseries = timeseries.loc[first_valid:last_valid]
            print("timeseries[0:3] AFTER = " + str(timeseries[0:3]))

            print("non-nan len(timeseries) = " + str(len(timeseries)))
            print("len(timeseries.values) = " + str(len(timeseries.values)))
            print(timeseries.values[-1])
            print("----------------------")
            if len(timeseries) > 0:
                train_ts.append(
                    to_dict(
                        target_values=timeseries.values,
                        start=time_index[0],
                        cat=[i],
                        item_id=i,
                    )
                )
            timeseries_out = df_out['m' + str(i)][3:]
            print("total len(timeseries_out) = " + str(len(timeseries_out)))
            print("timeseries_out = " + str(timeseries_out))
            # print(type(timeseries.values))
            # print(type(timeseries_out.values))
            for j in range(24):
                print("j+1 = " + str(j+1))
                extended_ts = np.array(list(timeseries.values) + list(timeseries_out.values[:j+1]))
                print("len(timeseries.values) = " + str(len(timeseries.values)))
                print("len(timeseries_out.values[:j+1]) = " + str(len(timeseries_out.values[:j+1])))
                print("len(extended_ts) = " + str(len(extended_ts)))
                test_ts.append(
                    to_dict(
                        target_values=extended_ts,
                        start=time_index[0],
                        cat=[i],
                        item_id=i,
                    )
                )  
                      
                    
        assert len(train_ts) == ds_info.num_series
        assert len(test_ts) == ds_info.num_series * ds_info.rolling_evaluations
        save_to_file(train_file, train_ts) 
        save_to_file(test_file, test_ts)

    '''

    # the last date seen during training
    ts_index = timeseries[0].index
    # print(timeseries[0].index)
    # print(len(ts_index))
    # training_end = ts_index[int(len(ts_index) * (8 / 10))]
    training_end = ts_index[-7 * 24 - 1]
    
    train_ts = []
    for cat, ts in enumerate(timeseries):
        sliced_ts = ts[:training_end]
        if len(sliced_ts) > 0:
            train_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                    item_id=cat,
                )
            )


    assert len(train_ts) == ds_info.num_series

    save_to_file(train_file, train_ts)

    # time of the first prediction
    prediction_dates = [
        frequency_add(training_end, i * ds_info.prediction_length)
        for i in range(ds_info.rolling_evaluations)
    ]

    test_ts = []
    t = 0
    for prediction_start_date in prediction_dates:
        print("t = " + str(t))
        t += 1
        for cat, ts in enumerate(timeseries):
            # print(prediction_start_date)
            prediction_end_date = frequency_add(
                prediction_start_date, ds_info.prediction_length
            )
            sliced_ts = ts[:prediction_end_date]
            test_ts.append(
                to_dict(
                    target_values=sliced_ts.values,
                    start=sliced_ts.index[0],
                    cat=[cat],
                    item_id=cat,
                )
            )

    assert len(test_ts) == ds_info.num_series * ds_info.rolling_evaluations

    save_to_file(test_file, test_ts)'''

if __name__ == "__main__":
    # dataset_name = "electricity_nbeats_last7days_predlen1"
    dataset_name = "tourism_monthly_predlen1"
    generate_lstnet_dataset(Path('.'), 
                            "tourism_monthly_predlen1", 
                            None,
                            csv_in='./tourism_alt/monthly_in.csv', 
                            csv_out='./tourism_alt/monthly_oos.csv')
    '''dataset_path = Path('.')
    dataset = load_datasets(
        metadata=dataset_path,
        train=dataset_path / dataset_name / "train",
        test=dataset_path / dataset_name / "test",
    )
    # dataset = get_dataset("electricity_nbeats_last7days", Path('.'),regenerate=False)
    train_ds = list(dataset.train)
    test_ds = list(dataset.test)
    print("len(train_ds) = " + str(len(train_ds)))
    print("len(test_ds) = " + str(len(test_ds)))
    print("len(train_ds[0]['target'] = " + str(len(train_ds[0]['target'])))
    print("len(test_ds[0]['target'] = " + str(len(test_ds[0]['target'])))
    print("len(train_ds[-1]['target'] = " + str(len(train_ds[-1]['target'])))
    print("len(test_ds[-1]['target'] = " + str(len(test_ds[-1]['target'])))
    print(train_ds[0])
    print(train_ds[-1])
    print(test_ds[0])
    print(test_ds[-1])'''