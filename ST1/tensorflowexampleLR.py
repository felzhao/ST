__author__ = 'felzhao'
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv(
    "https://raw.githubusercontent.com/sqy941013/learnmachinelearning/master/california_housing_train.csv",
    sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
   """ѵ��һ�������������Իع�ģ��

   ����:
      features: ������Pandas DataFrame����
     targets: Ŀ���Pandas DataFrame����
      batch_size: ����ģ�͵����δ�С
     shuffle: �Ƿ�����ݽ����������
      num_epochs: ָ����������������ΪNone������ѭ��
    ����:
      ��һ�����ݵ������ͱ�ǩ��ɵ�Ԫ��
    """

    # ��Pandas����ת��ΪNumpy�ֵ�
    features = {key: np.array(value) for key, value in dict(features).items()}

    # �������ݼ����������κ��ظ�
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # ������һ������
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """ѵ��һ�������������Իع�ģ��

    ����:
      learning_rate: �����ͣ�����ѧϰ������
      steps: �������ͣ�ѵ�����ܲ���
      batch_size:�������ͣ����δ�С
      input_feature: �ַ�����california_housing_dataframe����Ϊ���������ĵ�һ��
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # ����������
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # �������뺯��
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # �������Իع����
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # ����ģ�͵Ļ�ͼ����
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # ѵ��ģ��
    print("Training model...")

    print("RMSE (on training data):")

    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # ����Ԥ��
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # ������ʧ
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # �����ʧֵ
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))

        root_mean_squared_errors.append(root_mean_squared_error)
        # ����Ȩ�غ�ƫ��
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")


    # ������ʧͼ
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # ����������ݱ�
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    plt.show()

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)



train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=5
)
