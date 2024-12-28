import tensorflow as tf

class LabelSmoothedCE(tf.keras.losses.Loss):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps, n_predictions):
        """
        Init.

        Args:
            eps (float): Smoothing coefficient. 
            n_predictions (int): Number of predictions expected per
            datapoint, or length of the predicted sequence.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.indices = tf.range(n_predictions)  # (n_predictions)

    def call(self, y_true, y_pred): #lengths):
        lengths = tf.cast(y_true["lengths"], dtype=tf.int32)
        y_true = y_true["moves"][:, 1:]
        #print("true:",y_true)
        #print("pred:",y_pred)
        """
        Forward prop.

        Args:
            y_true (tf.Tensor): The actual targets, of size (N, n_predictions).
            y_pred (tf.Tensor): The predicted probabilities,
            of size (N, n_predictions, vocab_size).
            lengths (tf.Tensor): The true lengths of the
            prediction sequences, not including special tokens, of size (N, 1).

        Returns:
            tf.Tensor: The mean label-smoothed cross-entropy loss, a scalar.
        """
        # Remove pad positions based on lengths
        lengths = tf.squeeze(lengths)  # Shape (N,)
    
         # Create mask based on lengths
        mask = self.indices < lengths[:, None]

        predicted = tf.boolean_mask(y_pred, mask)  # (sum(lengths), vocab_size)
        targets = tf.boolean_mask(y_true, mask)  # (sum(lengths))

        target_vector = tf.one_hot(tf.cast(targets, dtype=tf.int32), depth=tf.shape(predicted)[1])

        # Apply label smoothing
        target_vector = target_vector * (1.0 - self.eps) + (self.eps / tf.cast(tf.shape(predicted)[1], tf.float32))


        # Compute smoothed cross-entropy loss
        loss = -tf.reduce_sum(target_vector * tf.nn.log_softmax(predicted), axis=1)  # (sum(lengths))

        # Compute mean loss
        loss = tf.reduce_mean(loss)

        return loss
