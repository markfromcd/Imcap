import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:
        # Now we will define image and word embedding, decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        # with the models hidden size
        self.image_embedding = tf.keras.layers.Dense(hidden_size)

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size ,self.hidden_size)

        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(self.hidden_size, return_sequences = True)

        # Define classification layer (LOGIT OUTPUT)
        self.dense = tf.keras.layers.Dense(self.hidden_size, activation="relu")
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        embeded_imgaes = self.image_embedding(encoded_images)
        #print(embeded_imgaes.shape)
        #captions = tf.reshape(captions, [-1, self.window_size])
        embeded_captions = self.embedding(captions)
        #print(embeded_captions)
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        decoder = self.decoder(embeded_captions, initial_state = embeded_imgaes)
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        logits = self.dense(decoder)
        logits = self.classifier(logits)
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size, activation='relu')

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size, False)

        # Define classification layer (logits)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        embeded_imgaes = self.image_embedding(tf.expand_dims(encoded_images, axis=1))
        #print(embeded_imgaes.shape)
        #captions = tf.reshape(captions, [-1, self.window_size])
        embeded_captions = self.encoding(captions)
        #print(embeded_captions)
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        decoder = self.decoder(embeded_captions, embeded_imgaes)
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        probs = self.classifier(decoder)
        return probs
