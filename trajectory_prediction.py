from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    BatchNormalization
)
from sklearn.preprocessing import MinMaxScaler

# Frequency for recordings at all files == 25 hz

TIMESTEP_LEN = 20  # how long of a preceeding timestep to collect for RNN
LOOK_BACK = 10  # (Frames) how far into the future are we trying to predict?
EPOCHS = 50
BATCH_SIZE = 128
TEST_PCT = 10  # %
features_scaler = MinMaxScaler()

assert LOOK_BACK >= 0  # !!!


def get_recordings_df(recordings_paths):
    return pd.concat(pd.read_csv(r, index_col='id') for r in recordings_paths)


def get_biggest_recording_id(recordings_dfs, reference_attr='numVehicles'):
    df = recordings_dfs

    max_duration = df[reference_attr].max()
    idx = df[df[reference_attr] == max_duration].index[0]

    return str(idx)


def get_model(x_train):
    model = Sequential()

    model.add(LSTM(256, batch_input_shape=(None, x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def split_target(df):
    y = df.pop('target').shift(-LOOK_BACK, fill_value=0)

    return df.to_numpy(), y.to_numpy()


def make_timesteps(features, targets):
    timesteps = []

    for start in range(features.shape[0]-TIMESTEP_LEN-LOOK_BACK):
        end = start + TIMESTEP_LEN

        timestep = features[start:end+1]
        target = targets[end]

        timesteps.append((timestep, target))

    np.random.shuffle(timesteps)

    X = np.array([X for X, _ in timesteps])
    y = np.array([y for _, y in timesteps])

    return X, y


# def get_surroundings():


def prepare_data(main_df):
    def prepare_data_aux(df):
        df.set_index('frame', inplace=True)
        df.sort_index(inplace=True)

        x_train, y_train = split_target(df)
        x_train, y_train = make_timesteps(x_train, y_train)

        return x_train, y_train

    # TODO: join velocities from surrounding vehicles
    main_df = main_df[[c for c in main_df.columns if not c.endswith('Id')]]
    return map(
        np.concatenate,
        zip(*[prepare_data_aux(d) for _, d in
              main_df.groupby('id', as_index=False)])
    )


def get_mock(rows=10**4):
    sequence = list(range(rows))

    return pd.DataFrame(
        dict(number=sequence, target=sequence),
        dtype='float32'
    )


data_folder = Path("./highD-dataset-v1.0/data")

recordings_paths = data_folder.glob('*_recordingMeta.csv')
recordings = get_recordings_df(recordings_paths)
recording_idx = get_biggest_recording_id(recordings)

# Index must be sequential on time
main_df = pd.read_csv(data_folder / f'{recording_idx}_tracks.csv')

# TODO: filter by vehicle type
# TODO: add another output to the model to the yVelocity
# TODO: find solution to the big look back value, limited by numFrames of recording
# TODO: use all recordings
main_df['target'] = main_df['xVelocity']  # target will be shifted in split_target()

# Train and transform scaler only at features
target = main_df.pop('target')
features_scaler.fit_transform(main_df)
main_df['target'] = target

validation_size = int(TEST_PCT/100 * main_df.shape[0])
validation_df = main_df.iloc[-validation_size:]
main_df = main_df.iloc[:-validation_size]

x_train, y_train = prepare_data(main_df)

# Transform scaler only at features
target = validation_df.pop('target')
features_scaler.transform(validation_df)
validation_df['target'] = target

x_validation, y_validation = prepare_data(validation_df)
assert x_validation.shape[0] != 0, "Validation empty!"

model = get_model(x_train)

model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_validation, y_validation)
)
