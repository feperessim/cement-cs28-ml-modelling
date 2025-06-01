from tensorflow.keras import layers
from tensorflow import keras

# source https://keras.io/examples/audio/transformer_asr/


class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate=0.1,
        activation="relu",
    ):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation=activation),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,  # embed_dim - num of features
        time_steps=7,
        num_head=2,
        num_feed_forward=128,  # pointwise dim
        num_layers_enc=4,
        dropout_rate=0.1,
        activation="relu",
    ):
        super().__init__()
        self.numlayers_enc = num_layers_enc
        self.enc_input = layers.Input((time_steps, num_hid))
        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(
                    num_hid, num_head, num_feed_forward, dropout_rate, activation
                )
                for _ in range(num_layers_enc)
            ]
        )
        self.GlobalAveragePooling1D = layers.GlobalAveragePooling1D(
            data_format="channels_last"
        )
        self.out = layers.Dense(units=1, activation="linear")

    def call(self, inputs):
        # x =  Time2Vector(x.shape[-1])
        x = self.encoder(inputs)
        x = self.GlobalAveragePooling1D(x)
        y = self.out(x)
        return y
