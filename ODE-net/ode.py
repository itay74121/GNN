import tensorflow as tf
from tfdiffeq.models import ODENet
import numpy as np


def main():
    model = tf.keras.Sequential([
        ODENet(32, 32, augment_dim=0, time_dependent=True),
        tf.keras.layers.Reshape((1,1,32)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(5,activation='softmax')
    ])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=(100//10)*20,
    decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    model.compile(
    optimizer=optimizer, loss='categorical_crossentropy', metrics=["acc"],
    )
    x = np.random.random((3,2))
    y = np.random.random((3,5))
    model.fit(x=x,y=y,epochs=2,batch_size=1)
    # print(x)
    # print("##############################################")
    # print(model(x).numpy())
if __name__ == "__main__":
    main()
    
    
    