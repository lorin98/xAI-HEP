import tensorflow as tf

def spread(y_true, y_pred):
    # consider only the first column (i.e. pt, not eta)
    y_true = tf.gather(y_true, [0], axis=1)
    y_pred = tf.gather(y_pred, [0], axis=1)

    mask = tf.math.logical_and(y_true >= 7, y_true <= 13)
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    return tf.math.reduce_mean(tf.abs(masked_y_true - masked_y_pred))

def efficiency(y_true, y_pred):
    # consider only the first column (i.e. pt, not eta)
    y_true = tf.gather(y_true, [0], axis=1)
    y_pred = tf.gather(y_pred, [0], axis=1)

    den_mask = tf.math.logical_and(y_true >= 15, True)
    num_mask = tf.math.logical_and(y_true >= 15, y_pred >= 15)
    den = tf.boolean_mask(y_true, den_mask)
    num = tf.boolean_mask(y_pred, num_mask)
    if tf.size(den) == 0:
        return tf.constant(1.0, dtype=tf.float64)
    return tf.size(num)/tf.size(den)

def spread_pt(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = tf.math.logical_and(y_true >= 7, y_true <= 13)
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)

    return tf.math.reduce_mean(tf.abs(masked_y_true - masked_y_pred))

def efficiency_pt(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    den_mask = tf.math.logical_and(y_true >= 15, True)
    num_mask = tf.math.logical_and(y_true >= 15, y_pred >= 15)
    den = tf.boolean_mask(y_true, den_mask)
    num = tf.boolean_mask(y_pred, num_mask)
    if tf.size(den) == 0:
        return tf.constant(1.0, dtype=tf.float32)
    return tf.cast(tf.size(num)/tf.size(den), dtype=tf.float32)