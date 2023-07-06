"""
brief: Collection of custom callbacks
author: Steffen Schotthöfer
date: 26.08.2021
"""
import tensorflow as tf


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    # def on_test_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print("The average loss for epoch {} is {:7.2f}.".format(epoch, logs["loss"]))


class HaltWhenCallback(tf.keras.callbacks.Callback):
    def __init__(self, quantity, tol):
        """
        Should be used in conjunction with
        the saving criterion for the model; otherwise
        training will stop without saving the model with quantity <= tol
        """
        super(HaltWhenCallback, self).__init__()
        if type(quantity) == str:
            self.quantity = quantity
        else:
            raise TypeError('HaltWhen(quantity,tol); quantity must be a string for a monitored quantity')
        self.tol = tol

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            if logs.get(self.quantity) < self.tol:
                print('\n\n', self.quantity, ' has reached', logs.get(self.quantity), ' < = ', self.tol,
                      '. End Training.')
                self.model.stop_training = True
        else:
            pass


class LearningRateSchedulerWithWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, lr_schedule):
        super(LearningRateSchedulerWithWarmup, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.lr_schedule = lr_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = (epoch + 1) / self.warmup_epochs * self.lr_schedule(epoch)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        else:
            lr = self.lr_schedule(epoch)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        print("Current learning rate: " + str(tf.keras.backend.get_value(self.model.optimizer.lr)))
