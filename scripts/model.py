import keras
import os
import pickle


def split_input_target(sequence, sequence_target=None):
    """
    Function that takes a sequence as argument and separates the sequence between input and label. The input will be the vector
    except of the last entry of each row and the label will be everything except the first entry.

    arguments
    sequence: sequential data to be separated into input and label
    returns
    input and target sequence

    """

    # Input sequence should always be there
    input_seq = sequence[:, 0:-1, :]
    # Checks for group masking and random masking
    if sequence_target is not None:
        if sequence_target.any():
            target_seq = sequence_target[:, 1:, :]
        else:
            target_seq = sequence[:, 1:, :]
    else:
        target_seq = sequence[:, 1:, :]

    return input_seq, target_seq


class CLM:
    """

    Chemical Language Models for Training

    """

    def __init__(
        self,
        model_parameters: dict,  # Parameters of the model, saved in a dictionary
        mode: str,  # Mode either Train or Predict
        saving_path: str,
        pre_trained_model_path: str = None,
    ):

        super(CLM, self).__init__()

        self.saving_path = saving_path  # Saving path of the model
        self.model_parameters = model_parameters
        self.info_size = self.model_parameters["info_size"]
        self.mode = mode
        self.pre_trained_model_path = pre_trained_model_path

        # If model on predict mode, then batch size to one, so that we can sampled one-by-one. Stateful is True,
        # so information is remebered between batches.
        if self.mode == "Predict":
            stateful = True
            self.batch_size = 1
        else:
            stateful = False
            self.batch_size = self.model_parameters["batch_size_finetune"]
            self.learning_rate = self.model_parameters["learning_rate_finetune"]

        # Creation of the LSTM layers of the model. This is used if one wants to experiment with different layers and sizes.
        self.layers_lstm = []
        for layer_ix, hidden_units in enumerate(self.model_parameters["size_layers"]):
            self.lstm = keras.layers.LSTM(
                hidden_units,
                return_sequences=True,
                activation=self.model_parameters["lstm_activation"],
                recurrent_activation=self.model_parameters["lstm_recurrentactivation"],
                dropout=self.model_parameters["dropout_rate"],
                stateful=stateful,
                name=f"lstm{layer_ix}_layer",
            )
            self.layers_lstm.append(self.lstm)

        self.dense = keras.layers.TimeDistributed(
            keras.layers.Dense(
                self.model_parameters["info_size"],
                activation=self.model_parameters["dense_activation"],
                name="output_layer",
            )
        )

        self.epochs = self.model_parameters["n_epochs"]  # Epochs for training

        self.optimizer = keras.optimizers.get(self.model_parameters["optimizer_name"])
        self.loss = self.model_parameters["loss"]
        self.metrics = self.model_parameters["metric"]

    def call(self, inputs, training=None):
        """
        Calling of the model. See keras functional API for more information of it.
        """

        x_s = inputs

        for layer_lstm in self.layers_lstm:
            x_s = layer_lstm(x_s, training=training)

        x_out = self.dense(x_s)

        return x_out

    def fine_tune_model(self, xTrain, yTrain, xVal, yVal):
        """
        Calls the fine-tuning mode of the model. If mode when calling the function is 'predict', then error when
        calling fine-tuning.
        arguments
        xTrain, yTrain: Training set. y is the label.
        xVal, yVal: Validation set. If none, then split mode.
        returns
        trained model
        history of the training process
        """

        my_callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.saving_path, "all-epochs", "model-{epoch:02d}.keras"),
                monitor="val_loss",
                save_best_only=False, 
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", min_delta=0.0001, patience=5
            ),
        ]

        inputs = keras.layers.Input((None, xTrain.shape[2]))
        model = keras.models.Model(inputs=[inputs], outputs=self.call(inputs))

        # Define the optimizer with gradient clipping
        updated_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0
        )

        # Calling of the weights of the (in the past) trained model
        trained_model = keras.models.load_model(
            os.path.join(self.pre_trained_model_path, "model.h5")
        )

        model.set_weights(trained_model.get_weights())

        model.compile(
            optimizer=updated_optimizer, loss=self.loss, metrics=[self.metrics]
        )

        # Training of the model
        history = model.fit(
            xTrain,
            yTrain,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(xVal, yVal),
            verbose=1,
            callbacks=my_callbacks,
        )

        # Keras only allows saving of checkpoints during training in .keras format 
        # So we also save the final epoch's checkpoint to .h5 format
        model.save(os.path.join(self.saving_path, "model.h5"))

        # Save the training history for loss curve plotting.
        with open(
            os.path.join(self.saving_path, "training_history.pkl"), "wb"
        ) as history_file:
            pickle.dump(history.history, history_file)

        return model, history

    def predict_model(self):
        """
        Prediction mode of the model. If mode is Train, then error message.
        Model is build and weights are based on the saved training model.
        arguments: None
        returns
        model ready for prediction
        """

        if self.mode != "Predict":
            raise ValueError(
                'Mode of the Model is still in "Train" or "Finetune". Call "Predict" mode.'
            )

        inputs = keras.layers.Input(batch_shape=(self.batch_size, None, self.info_size))
        model = keras.models.Model(inputs=[inputs], outputs=self.call(inputs))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metrics])

        # Calling of the weights of the (in the past) trained model
        if ".keras" in self.saving_path:
            model_path = self.saving_path
        else:
            model_path = os.path.join(self.saving_path, "model.h5")

        trained_model = keras.models.load_model(model_path)
        model.set_weights(trained_model.get_weights())

        return model
