from ...core import BaseModule


class LMModule(BaseModule):
    """ Application class of language modeling (LM). """

    def fit_from_tfrecords(
        self,
        batch_size=32,
        learning_rate=5e-5,
        target_steps=None,
        total_steps=1000000,
        warmup_ratio=0.01,
        print_per_secs=0.1,
        save_per_steps=10000,
        tfrecords_files=None,
        n_jobs=3,
        **kwargs,
    ):
        """ Training the model using TFRecords.

        Args:
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            tfrecords_files: list. A list object of string defining TFRecords
              files to read.
            n_jobs: int. Number of threads in reading TFRecords files.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        """
        super().fit_from_tfrecords(
            batch_size,
            learning_rate,
            target_steps,
            total_steps,
            warmup_ratio,
            print_per_secs,
            save_per_steps,
            tfrecords_files,
            n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X=None, y=None, sample_weight=None, X_tokenized=None,
        batch_size=32,
        learning_rate=5e-5,
        target_steps=None,
        total_steps=1000000,
        warmup_ratio=0.01,
        print_per_secs=0.1,
        save_per_steps=10000,
        **kwargs,
    ):
        """ Training the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            y: list. A list object consisting labels.
            sample_weight: list. A list object of float-convertable values.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        """
        super().fit(
            X, y, sample_weight, X_tokenized,
            batch_size,
            learning_rate,
            target_steps,
            total_steps,
            warmup_ratio,
            print_per_secs,
            save_per_steps,
            **kwargs,
        )

    def score(self, *args, **kwargs):
        raise AttributeError("`score` method is not supported for unsupervised language modeling (LM) modules.")
