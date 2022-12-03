import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_parquet("data/fhv_tripdata_2021-01.parquet")
print(df.head())

print(len(df))

df['duration'] = df.dropOff_datetime - df.pickup_datetime
df['duration'] = df.duration.dt.total_seconds() / 60

print(f"January : {df.duration.mean()}")

df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

categorical = ['PUlocationID', 'DOlocationID']

df[categorical] = df[categorical].fillna(-1).astype('int')

df[categorical] = df[categorical].astype('str')

train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

#X_train.shape
y_train = df.duration.values

len(dv.feature_names_)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

print(mean_squared_error(y_train, y_pred, squared=False))

categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df_val = read_data('data/fhv_tripdata_2021-02.parquet')

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = lr.predict(X_val)
y_val = df_val.duration.values
print(mean_squared_error(y_val, y_pred, squared=False))





