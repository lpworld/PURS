import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class Model(object):
    def __init__(self, user_count, item_count, batch_size):
        hidden_size = 128
        long_memory_window = 10
        short_memory_window = 3
        
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, long_memory_window]) # [B, T]
        self.lr = tf.placeholder(tf.float64, [])
        
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size // 2])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size // 2])
        user_b = tf.get_variable("user_b", [user_count], initializer=tf.constant_initializer(0.0),)
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))

        item_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(user_emb_w, self.u),
            ], axis=1)
        item_b = tf.gather(item_b, self.i)
        user_b = tf.gather(user_b, self.u)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,0], [batch_size, long_memory_window])),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, long_memory_window, 1]),
            ], axis=2)
        unexp_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,long_memory_window-short_memory_window], [batch_size, short_memory_window])),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, short_memory_window, 1]),
            ], axis=2)
        h_long_emb = tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,0], [batch_size, long_memory_window]))
        h_short_emb = tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,long_memory_window-short_memory_window], [batch_size, short_memory_window]))

        # Long-Short-Term User Preference
        #with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
        long_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32)
        long_preference, _ = self.seq_attention(long_output, hidden_size, long_memory_window)
        long_preference = tf.nn.dropout(long_preference, 0.1)
        #short_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=unexp_emb, dtype=tf.float32)
        #short_preference, _ = self.seq_attention(short_output, hidden_size, long_memory_window)
        #short_preference = tf.nn.dropout(short_preference, 0.1)

        #Combine Long-Short-Term-User-Preferences
        concat = tf.concat([long_preference, item_emb], axis=1)
        concat = tf.layers.batch_normalization(inputs=concat)
        concat = tf.layers.dense(concat, 80, activation=tf.nn.sigmoid, name='f1')
        concat = tf.layers.dense(concat, 40, activation=tf.nn.sigmoid, name='f2')
        concat = tf.layers.dense(concat, 1, activation=None, name='f3')
        concat = tf.reshape(concat, [-1])

        #Personalized & Contextualized Unexpected Factor
        unexp_factor = self.unexp_attention(item_emb, unexp_emb, [long_memory_window]*batch_size)
        unexp_factor = tf.layers.batch_normalization(inputs = unexp_factor)
        unexp_factor = tf.reshape(unexp_factor, [-1, hidden_size])
        unexp_factor = tf.layers.dense(unexp_factor, hidden_size)
        unexp_factor = tf.layers.dense(unexp_factor, 1, activation=None)
        #If we choose to use binary values
        #unexp_gate = tf.to_float(tf.reshape(unexp_gate, [-1]) > 0.5)
        unexp_factor = tf.reshape(unexp_factor, [-1])

        #Unexpectedness (with clustering of user interests)
        self.center = self.mean_shift(h_long_emb)
        unexp = tf.reduce_mean(self.center, axis=1)
        unexp = tf.norm(unexp-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1)
        self.unexp = unexp
        unexp = tf.exp(-1.0*unexp) * unexp #Unexpected Activation Function
        unexp = tf.stop_gradient(unexp)

        #Relevance (for future exploration)
        relevance = tf.reduce_mean(h_long_emb, axis=1)
        relevance = tf.norm(relevance-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1)

        #Annoyance/Diversification (for future exploration)
        annoyance = tf.reduce_mean(h_short_emb, axis=1)
        annoyance = tf.norm(annoyance-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1)

        #Estmation of user preference by combing different components
        self.logits = item_b + concat + user_b + unexp_factor*unexp # [B]exp
        self.score = tf.sigmoid(self.logits)

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.y))
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
                self.lr: lr,
                })
        return loss

    def test(self, sess, uij):
        score, unexp = sess.run([self.score, self.unexp], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
                })
        return score, uij[3], uij[0], uij[2], unexp

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    def extract_axis_1(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def seq_attention(self, inputs, hidden_size, attention_size):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
        for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
    
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
            attention_size: Linear size of the Attention weights.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
        """
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas

    def unexp_attention(self, querys, keys, keys_id):
        """
        Same Attention as in the DIN model
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of clicked creativeid for each sample)
        keys_id:     [Batchsize, max_seq_len]
        """
        querys = tf.expand_dims(querys, 1)
        keys_length = tf.shape(keys)[1] # padded_dim
        embedding_size = querys.get_shape().as_list()[-1]
        keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
        querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])

        net = tf.concat([keys, keys - querys, querys, keys*querys], axis=-1)
        for units in [32,16]:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        # shape(batch_size, max_seq_len, 1)
        outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  #shape(batch_size, 1, max_seq_len)
        scores = outputs
        scores = scores / (embedding_size ** 0.5)       # scale
        scores = tf.nn.softmax(scores)
        outputs = tf.matmul(scores, keys)    #(batch_size, 1, embedding_size)
        outputs = tf.reduce_sum(outputs, 1, name="unexp_embedding")   #(batch_size, embedding_size)
        return outputs

    def mean_shift(self, input_X, window_radius=0.2):
        X1 = tf.expand_dims(tf.transpose(input_X, perm=[0,2,1]), 1)
        X2 = tf.expand_dims(input_X, 1)
        C = input_X
        def _mean_shift_step(C):
            C = tf.expand_dims(C, 3)
            Y = tf.reduce_sum(tf.pow((C - X1) / window_radius, 2), axis=2)
            gY = tf.exp(-Y)
            num = tf.reduce_sum(tf.expand_dims(gY, 3) * X2, axis=2)
            denom = tf.reduce_sum(gY, axis=2, keep_dims=True)
            C = num / denom
            return C
        def _mean_shift(i, C, max_diff):
            new_C = _mean_shift_step(C)
            max_diff = tf.reshape(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(new_C - C, 2), axis=1))), [])
            return i + 1, new_C, max_diff 
        def _cond(i, C, max_diff):
            return max_diff > 1e-5
        n_updates, C , max_diff = tf.while_loop(cond=_cond, body=_mean_shift, loop_vars=(tf.constant(0), C, tf.constant(1e10)))
        return C
