import tensorflow as tf2
tf1 = tf2.compat.v1
import tensorflow as tf
from pylab import*
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.io.wavfile as wav
import librosa
from python_speech_features import mfcc
tf1.disable_eager_execution()

def def_wav_read_mfcc(file_name):
    try:
        fs, audio = wav.read(file_name)
        processed_audio = mfcc(audio, samplerate=fs, nfft = 2048)
    except ValueError:
        audio, fs = librosa.load(file_name)
        processed_audio = mfcc(audio, samplerate=fs, nfft = 2048)
    return processed_audio

# 训练数据的文件夹
file_path = r'../Dataset/audio/1/'
# 路径拼接
data = [os.path.join(file_path,i) for i in os.listdir(file_path)]
# 定义每个文件的标签
label = [[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[0],[1],[0],[0],[1],[0],[1],[1],[0],[0],[1],[0],[1],[0],[1],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[1],[1],[0],[0],[1],[1],[0],[1],[1],[0],[1],[0],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[1],[0],[1],[1],[1],[0],[1],[1],[1],[0],[0],[1],[0],[1],[0],[1],[1],[1],[0],[1],[0],[0],[0],[0],[0],[1],[1],[1],[0],[1],[0],[1],[0],[0],[0],[1],[1],[1],[1],[0],[0],[0],[0],[1],[0],[0],[1],[0],[0],[1],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[0],[1],[1],[0],[1],[1],[0],[1],[1],[1],[1],[1],[1],[1],[1],[0],[1],[1],[0],[1],[1],[1],[0],[0],[1],[1],[0],[1],[1],[1],[1],[0],[1],[1],[1],[0],[0],[0],[1],[0],[1],[0],[1],[1],[0],[0],[0],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[0],[1],[0],[0],[0],[0]]

def get_batch_for_train(i):
    Wav = []
    for j in range(i,198+i):
        wav = gf.def_wav_read_mfcc(data[j])
        wav1 = wav[:5000]
        Wav.append(wav1)
        print("Conversion to MFCC：",j+1)
    label_for_train = label[i:i+198]
    return Wav, label_for_train
Wav,label_for_train = get_batch_for_train(0)
n_inputs = 13
max_time = 5000
lstm_size = 78
n_classes = 1
batch_size = 20
n_batch = len(Wav) // batch_size
nums_samples = len(Wav)
n = 5000

for i in range(len(Wav)):
    scaler = MinMaxScaler(feature_range=(0, 1))
    Wav[i] = scaler.fit_transform(Wav[i])
# 生成的 Wav[] 里面放的是 array(15000×13)因此用循环将向量转换成列表
for i in range(len(Wav)):
    Wav[i] = Wav[i].tolist()
Wav_tensor = tf1.convert_to_tensor(Wav)
label_tensor = tf1.convert_to_tensor(label_for_train)
print("Success construct Wav_tensor")

def get_batch(data, label, batch_size):
    input_queue = tf1.data.Dataset.from_tensor_slices(data).repeat()
    input_queue_y = tf1.data.Dataset.from_tensor_slices(label).repeat()
    x_batch = input_queue.batch(batch_size)
    y_batch = input_queue_y.batch(batch_size)
    batch_x = tf.compat.v1.data.make_one_shot_iterator(x_batch)
    x_batch = batch_x.get_next()
    batch_y = tf.compat.v1.data.make_one_shot_iterator(y_batch)
    y_batch = batch_y.get_next()
    return x_batch, y_batch

x_batch, y_batch = get_batch(Wav_tensor, label_tensor, batch_size)
x_batch_test, y_batch_test = get_batch(Wav_tensor, label_tensor, 100)
x_batch = tf1.convert_to_tensor(x_batch)
y_batch = tf1.convert_to_tensor(y_batch)

x = tf1.placeholder(tf1.float32, [None, 5000, 13])
y = tf1.placeholder(tf1.float32, [None, 1])

def lstm_model(X, weights, biases):
    inputs = tf1.reshape(X, [-1, max_time, n_inputs])
    lstm_cell_1 = tf.keras.layers.LSTMCell(lstm_size)
    outputs_1, final_state_1= tf1.nn.dynamic_rnn(lstm_cell_1, inputs, dtype=tf1.float32)
    lstm_cell_2 = tf.keras.layers.LSTMCell(lstm_size)
    outputs_2, final_state_2= tf1.nn.dynamic_rnn(lstm_cell_2, outputs_1, dtype=tf1.float32)
    lstm_cell_3 = tf.keras.layers.LSTMCell(lstm_size)
    outputs_3, final_state_3= tf1.nn.dynamic_rnn(lstm_cell_3, outputs_2, dtype=tf1.float32)
    lstm_cell_4 = tf.keras.layers.LSTMCell(13)
    outputs, final_state= tf1.nn.dynamic_rnn(lstm_cell_4, outputs_3, dtype=tf1.float32)
    result = tf.nn.sigmoid(tf1.matmul(final_state[0], weights) + biases)
    return result

weights = tf1.Variable(tf1.truncated_normal([13, n_classes], stddev=0.1))
biases = tf1.Variable(tf1.constant(0.1, shape=[n_classes]))
prediction = lstm_model(x,weights,biases)
cross_entropy = tf1.reduce_mean(tf1.square(y - prediction),name ="cross_entropy" )
train_step = tf1.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf1.equal(y,tf.round(prediction))
accuracy = tf1.reduce_mean(tf1.cast(correct_prediction,tf1.float32),name = "accuracy")
init = tf1.global_variables_initializer()
config = tf1.ConfigProto()
config.gpu_options.allocator_type = "BFC"


saver = tf1.train.Saver()
with tf1.Session() as sess:
    with tf1.device('/gpu:0'):
        sess.run(init)
        loss = []
        checkpoint_steps = 100
        for i in range(600):
            x_batch_data = sess.run(x_batch)
            y_batch_data = sess.run(y_batch)
            x_batch_test_data = sess.run(x_batch_test)
            y_batch_test_data = sess.run(y_batch_test)
            sess.run(train_step, feed_dict={x: x_batch_data, y: y_batch_data})
            pred_X1 = sess.run(prediction, feed_dict={x: x_batch_data})
            pred_X1 = pred_X1[0]
            if pred_X1 >= 0.5:
                print("This sound is True.")
            else:
                print("This sound is False.")
            y_batch_data = y_batch_data[0]
            if y_batch_data[0] >= 0.5:
                y_r = True
            else:
                y_r = False
            print("prediction",i,":",prediction,";  ","real value：",y_batch_data,";  ","test result:",pred_X1)
            print("test results:",pred_X1,";  ","real value:",y_batch_data)
            if (i + 1) % 10 == 0:
                cross_entropy_new = sess.run(cross_entropy,feed_dict={x: x_batch_test_data, y: y_batch_test_data})
                accurace = sess.run(accuracy,feed_dict={x: x_batch_test_data, y: y_batch_test_data})
                loss.append(cross_entropy_new)
                print("accurace:",cross_entropy_new,accurace,"Iteration",i+1)

            if (i + 1) % checkpoint_steps == 0:
                saver.save(sess, "save1/model.ckpt", global_step=i + 1)
                print("Success save.")
    k = len(loss)
    t = []
    for i in range(k):
        t.append(i)
    print(t)
    plt.plot(t, loss, 'k-', label='Data', linewidth=2)
    font1 = {'size': 18}
    plt.legend(loc=4, prop=font1)
    plt.xlabel(u'Iteration', size=24)
    plt.ylabel(u'Loss', size=24)
    plt.show()
    saver.save(sess, "save1/model.ckpt")

