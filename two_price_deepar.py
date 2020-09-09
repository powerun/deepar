
import tensorflow as tf
import tensorflow_probability as tfp
import first_price_model
from tensorflow.keras import layers, callbacks
import os


class DeepAR(tf.keras.Model):
    """
    DeepAR
    """
    def __init__(self, lstm_units=32, initial_state=None):
        super(DeepAR, self).__init__()
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm2 = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.lstm3 = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        # self.dense_mu = tf.keras.layers.Dense(1, activation='softplus')
        self.dense_mu = tf.keras.layers.Dense(1)
        self.dense_sigma = tf.keras.layers.Dense(1, activation='softplus')
        self.initial_state = initial_state

    def call(self, inputs, **kwargs):
        outputs, state_h, state_c = self.lstm(inputs, initial_state=self.initial_state)
        outputs, state_h, state_c = self.lstm2(outputs)
        outputs, state_h, state_c = self.lstm3(outputs)
        mu = self.dense_mu(outputs)
        sigma = self.dense_sigma(outputs)
        state = [state_h, state_c]
        return [mu, sigma]

    # def compute_output_shape(self, input_shape):
    #     shape = tf.TensorShape(input_shape).as_list()
    #     shape[-1] = self.num_classes
    #     return tf.TensorShape(shape)


def log_gaussian_loss(y_true, y_pred):
    mu, sigma = y_pred[0], y_pred[1]
    y_true = tf.expand_dims(y_true, axis=2)
    # return -tf.reduce_sum(tfp.distributions.NegativeBinomial(1/sigma, 1/(1 + sigma*mu)).log_prob(y_true))
    return -tf.reduce_sum(tfp.distributions.Normal(loc=mu, scale=sigma).log_prob(y_true))


def deepar_model(day_num, feature_size):
    input_x = tf.keras.Input(shape=(day_num, feature_size))
    outputs, state_h, state_c = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)(input_x, initial_state=None)
    outputs, state_h, state_c = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)(outputs)
    output, state_h, state_c = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)(outputs)
    mu = tf.keras.layers.Dense(1)(output)
    sigma = tf.keras.layers.Dense(1, activation='softplus')(output)
    model = tf.keras.Model([input_x], [mu, sigma])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=log_gaussian_loss, metrics=['mape', 'mse'])
    model.summary()
    return model


def train_test():
    x, y, test_x, test_y = first_price_model.generate_data3()
    print(x.shape)
    print(y.shape)
    print(test_x.shape)
    print(test_y.shape)
    train_data1 = tf.data.Dataset.from_tensor_slices((x, y))
    train_data = train_data1.cache().shuffle(10000).batch(32).repeat()
    test_data1 = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_data = test_data1.cache().shuffle(10000).batch(128).repeat()
    model = DeepAR()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=log_gaussian_loss, metrics=['mape', 'mse'])
    cp_callback = [
        callbacks.ModelCheckpoint("./model_price/model_price_best.ckpt", save_weights_only=True, save_best_only=True,
                                  monitor='val_loss'),
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9, verbose=1, min_lr=0.00005),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs"))]
    history = model.fit(train_data, epochs=5000, steps_per_epoch=1000,
                        validation_data=test_data, validation_steps=10, callbacks=cp_callback)
    # model.save('./model_price/model_5000_1000_price.h5')


if __name__ == '__main__':
    train_test()








