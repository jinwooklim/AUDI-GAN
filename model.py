# -*- coding: utf-8 -*-
import os
import time
import subprocess
import argparse
import numpy as np
import tensorflow as tf
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TF_SEED = tf.set_random_seed(111)

parser = argparse.ArgumentParser()
parser.add_argument('--isresume', action='store_true', dest='isresume', default=False)
parser.add_argument('--istest', action='store_true', dest='istest', default=False)
parser.add_argument('--istarget', action='store_true', dest='istarget', default=False)
parser.add_argument('--gen_A', action='store_true', dest='is_gen_A', default=False)
parser.add_argument('--gen_B', action='store_true', dest='is_gen_B', default=False)
parser.add_argument('--input_dir', action='store', dest='input_dir', default=False)
parser.add_argument('--input_data', action='store', dest='input_data', default=False)
parser.add_argument('--model_dir', action='store', dest='model_dir', default='./model/')
parser.add_argument('--checkpoint', action='store', dest='checkpoint', default=False)
FLAGS = parser.parse_args()

if not FLAGS.istest:
    NPY_FOLDER_A = './train/preprocessed_A_npy'
    NPY_FOLDER_B = './train/preprocessed_B_npy'
else:
    NPY_FOLDER_A = './test/test_preprocessed_A_npy'
    NPY_FOLDER_B = './test/test_preprocessed_B_npy'

VALID_NPY = './train/valid_npy'
VALID_MP3 = './train/valid_mp3'

INF_RESULT_NPY_OUTPUT_IS_A = "./result/result_inference_npy_output_is_A"
INF_RESULT_NPY_OUTPUT_IS_B = "./result/result_inference_npy_output_is_B"
INF_RESULT_MP3_OUTPUT_IS_A = "./result/result_inference_mp3_output_is_A"
INF_RESULT_MP3_OUTPUT_IS_B = "./result/result_inference_mp3_output_is_B"

if not os.path.exists(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)
if not os.path.exists(VALID_NPY):
    os.mkdir(VALID_NPY)
if not os.path.exists(VALID_MP3):
    os.mkdir(VALID_MP3)
if not os.path.exists(INF_RESULT_NPY_OUTPUT_IS_A):
    os.mkdir(INF_RESULT_NPY_OUTPUT_IS_A)
if not os.path.exists(INF_RESULT_NPY_OUTPUT_IS_B):
    os.mkdir(INF_RESULT_NPY_OUTPUT_IS_B)
if not os.path.exists(INF_RESULT_MP3_OUTPUT_IS_A):
    os.mkdir(INF_RESULT_MP3_OUTPUT_IS_A)
if not os.path.exists(INF_RESULT_MP3_OUTPUT_IS_B):
    os.mkdir(INF_RESULT_MP3_OUTPUT_IS_B)


#hyperparameters
learning_rate = 0.0001
if not FLAGS.istest:
    batch_size = 4
else:
    batch_size = 1
capacity = batch_size * 3
channel = 1
initializer = tf.truncated_normal_initializer(stddev=0.02)
iteration = 100000
NUM_THREADS = 2


def call_gen_audio_subprocess(batch_npy, iteration, genfrom):
    for batch_idx in range(batch_size):
        temp = batch_npy[batch_idx, :, :, :]
        filename_npy = "%s-%06d-%s.npy"%(genfrom, iteration, batch_idx)
        filename_mp3 = "%s-%06d-%s.mp3"%(genfrom, iteration, batch_idx)
        np.save(os.path.join(VALID_NPY, filename_npy), temp)
        subprocess.Popen("python gen_audio.py --input_dir=%s --input=%s --output_dir=%s --output=%s"
                %(VALID_NPY, filename_npy, VALID_MP3, filename_mp3), shell=True)


def make_convlayer(X, w_name, b_name, w_shape ):
    w = tf.get_variable(w_name, w_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(b_name, w_shape[3], initializer = tf.constant_initializer(0))
    layer = tf.nn.conv2d(input = X , filter = w, strides = [1,2,2,1] , padding = 'SAME') + b
    #layer = tf.contrib.layers.batch_norm(layer)
    layer = tf.nn.leaky_relu(layer)
    return layer


def make_deconvlayer(X, w_name, b_name, w_shape):
    w = tf.get_variable(w_name, w_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(b_name, w_shape[2], initializer = tf.constant_initializer(0))
    shape = tf.shape(X)
    out_shape = tf.stack([shape[0], shape[1]*2, shape[2]*2, tf.shape(w)[2]]) # ?, 32*2, 8*2 ,128
    layer = tf.nn.conv2d_transpose(value = X , filter = w, strides = [1,2,2,1] ,output_shape=out_shape, padding = 'SAME') + b
    #layer = tf.contrib.layers.batch_norm(layer)
    layer = tf.nn.leaky_relu(layer)
    return layer


def feature_loss(image, recons_image):
    # exp(x)+1 for inverse log1p
    image = tf.exp(image)-1
    recons_image = tf.exp(recons_image)-1

    log_offset = 1e-6
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 7600.0, 64

    # Get Image log mel-spectograms
    num_spectrogram_bins = image.shape[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, config.SAMPLEING_RATE, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(image, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(image.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    image_mel_spectrograms = tf.log(mel_spectrograms + log_offset)


    # Get Reconst_image log mel-spectogram
    num_spectrogram_bins = image.get_shape()[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, config.SAMPLEING_RATE, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(recons_image, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(recons_image.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    reconst_image_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    
    # Calculate feature loss mean_squeared
    loss_mel = tf.losses.mean_squared_error(image_mel_spectrograms, reconst_image_mel_spectrograms)
    return loss_mel


def discriminator(image, scope, reuse=False):
    with tf.variable_scope(scope):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        
        disc1 = make_convlayer(image,'w1','b1', [4,4,1,32])
        disc2 = make_convlayer(disc1,'w2','b2', [4,4,32,64])
        disc3 = make_convlayer(disc2,'w3','b3', [4,4,64,128])
        
        #fully connected layers
        disc_f1 =tf.reshape(disc3, shape=[-1, 128 * 16 * 128])
        disc_wf1 = tf.get_variable('wf1', [128 * 16 * 128, config.WIDTH], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disc_bf1 = tf.get_variable('bf1', [config.WIDTH], initializer = tf.constant_initializer(0))
        disc_f1 = tf.matmul(disc_f1,disc_wf1) + disc_bf1
        disc_f1 = tf.nn.leaky_relu(disc_f1)
        
        #output sigmoid layer
        disc_wf2 = tf.get_variable('wf2', [config.WIDTH,1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        disc_bf2 = tf.get_variable('bf2', [1], initializer = tf.constant_initializer(0))
        disc_f2 = tf.matmul(disc_f1,disc_wf2) + disc_bf2
        
        return disc_f2


def generator(image, scope, reuse=False):
    with tf.variable_scope(scope):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        #encoder layers
        enc_layer1 = make_convlayer(image,'enc_w1','enc_b1', [4,4,1,32])
        enc_layer2 = make_convlayer(enc_layer1,'enc_w2','enc_b2', [4,4,32,64])
        enc_layer3 = make_convlayer(enc_layer2,'enc_w3','enc_b3', [4,4,64,128])
        enc_final = make_convlayer(enc_layer3,'enc_w4','enc_b4', [4,4,128,256])
        
        #decoder layers 
        dec_layer4 = make_deconvlayer(enc_final,'dec_w4','dec_b4', [4,4,128,256])
        dec_layer3 = make_deconvlayer(dec_layer4,'dec_w3','dec_b3', [4,4,64,128])
        dec_layer2 = make_deconvlayer(dec_layer3,'dec_w2','dec_b2',[4,4,32,64])
        generated_image = make_deconvlayer(dec_layer2,'dec_w1','dec_b1',[4,4,1,32])

        return generated_image


def train():
    files_A = tf.convert_to_tensor([tf.convert_to_tensor(np.reshape(np.load(os.path.join(NPY_FOLDER_A, f)), 
        (config.HEIGHT, config.WIDTH, channel)), dtype=tf.float32) for f in os.listdir(NPY_FOLDER_A + '/')])
    files_B = tf.convert_to_tensor([tf.convert_to_tensor(np.reshape(np.load(os.path.join(NPY_FOLDER_B, f)), 
        (config.HEIGHT, config.WIDTH, channel)), dtype=tf.float32) for f in os.listdir(NPY_FOLDER_B + '/')])

    queue_A = tf.RandomShuffleQueue(capacity=capacity, 
        min_after_dequeue=int(0.5*capacity), 
        shapes=(config.HEIGHT, config.WIDTH, channel), 
        dtypes=tf.float32)

    queue_B = tf.RandomShuffleQueue(capacity=capacity, 
        min_after_dequeue=int(0.5*capacity), 
        shapes=(config.HEIGHT, config.WIDTH, channel), 
        dtypes=tf.float32)

    enqueue_op_A = queue_A.enqueue_many(files_A)
    enqueue_op_B = queue_B.enqueue_many(files_B)

    dequeue_op_A = queue_A.dequeue_many(batch_size)
    dequeue_op_B = queue_B.dequeue_many(batch_size)
    
    # image.height , image.width
    with tf.name_scope('input'):
        A = tf.placeholder(tf.float32,[None, config.HEIGHT, config.WIDTH, channel], name= 'A_data') # A
        B = tf.placeholder(tf.float32,[None, config.HEIGHT, config.WIDTH, channel], name= 'B_data') # B

    #scope_variables
    scope_gen_A = 'generator_A'
    scope_gen_B = 'generator_B'
    scope_disc_A = 'discriminator_A'
    scope_disc_B = 'discriminator_B'
    scope_disc_style = 'discriminator_style'

    # Generation & Discrimination
    with tf.name_scope('gen'):
        gen_AB = generator(A, scope_gen_B)
        gen_BA = generator(B, scope_gen_A)
    
    with tf.name_scope('recons'):
        recons_BAB = generator(gen_BA, scope_gen_B, reuse = True)
        recons_ABA = generator(gen_AB, scope_gen_A, reuse = True)

    with tf.name_scope('disc'): 
        disc_A_gen = discriminator(gen_BA, scope_disc_A)
        disc_B_gen = discriminator(gen_AB, scope_disc_B)

        disc_A_real = discriminator(A, scope_disc_A, reuse = True)
        disc_B_real = discriminator(B, scope_disc_B, reuse = True)

        disc_A_recons = discriminator(recons_ABA, scope_disc_A, reuse = True)
        disc_B_recons = discriminator(recons_BAB, scope_disc_B, reuse = True)

        disc_style_A = discriminator(A, scope_disc_style)
        disc_style_B = discriminator(B, scope_disc_style, reuse = True)
        disc_style_AB = discriminator(gen_AB, scope_disc_style, reuse = True)
        disc_style_BA = discriminator(gen_BA, scope_disc_style, reuse = True)
        disc_style_ABA = discriminator(recons_ABA, scope_disc_style, reuse = True)
        disc_style_BAB = discriminator(recons_BAB, scope_disc_style, reuse = True)

    with tf.name_scope('recons_loss'):
        loss_recons_A = tf.reduce_sum(tf.losses.mean_squared_error(A, recons_ABA))
        loss_recons_B = tf.reduce_sum(tf.losses.mean_squared_error(B, recons_BAB))
    
    with tf.name_scope('gen_loss'):
        # LSGAN version generator loss
        loss_gen_A = tf.reduce_sum(tf.square(disc_A_gen-1))/2
        loss_gen_B = tf.reduce_sum(tf.square(disc_B_gen-1))/2
        loss_total_gen_A = loss_gen_A + loss_recons_A + feature_loss(A, recons_ABA)
        loss_total_gen_B = loss_gen_B + loss_recons_B + feature_loss(B, recons_BAB)

        # Total generator loss
        loss_gen = loss_total_gen_A + loss_total_gen_B
    
    with tf.name_scope('disc_loss'):
        # LSGAN version discriminator loss
        loss_disc_A = tf.reduce_sum(tf.square(disc_A_real-1) + tf.square(disc_A_gen))/2
        loss_disc_B = tf.reduce_sum(tf.square(disc_B_real-1) + tf.square(disc_B_gen))/2
        
        # TODO for recons samples
        loss_disc_recons_A = tf.reduce_sum(tf.square(disc_A_real-1) + tf.square(disc_A_recons))/2
        loss_disc_recons_B = tf.reduce_sum(tf.square(disc_B_real-1) + tf.square(disc_B_recons))/2
        loss_total_disc_recons = loss_disc_recons_A + loss_disc_recons_B

        # TODO style discriminator loss
        loss_total_disc_style = tf.reduce_sum(tf.square(disc_style_A-1) + tf.square(disc_style_BA))/2 \
                                + tf.reduce_sum(tf.square(disc_style_B-1) + tf.square(disc_style_AB))/2 \
                                + tf.reduce_sum(tf.square(disc_style_A-1) + tf.square(disc_style_ABA))/2 \
                                + tf.reduce_sum(tf.square(disc_style_B-1) + tf.square(disc_style_BAB))/2

        # Total discriminator loss
        loss_disc = loss_disc_A + loss_disc_B + loss_total_disc_recons + loss_total_disc_style

    vars_dis = tf.trainable_variables(scope_disc_B) + tf.trainable_variables(scope_disc_A) + tf.trainable_variables(scope_disc_style)
    vars_gen = tf.trainable_variables(scope_gen_B) + tf.trainable_variables(scope_gen_A)

    with tf.name_scope('trainer'):
        disc_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7).minimize(loss_disc, var_list=vars_dis)
        gen_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7).minimize(loss_gen, var_list=vars_gen)

    tf.add_to_collection('generated_A', gen_BA)
    tf.add_to_collection('generated_B', gen_AB)
    
    qr_01 = tf.train.QueueRunner(queue_A, [enqueue_op_A] * NUM_THREADS)
    qr_02 = tf.train.QueueRunner(queue_B, [enqueue_op_B] * NUM_THREADS)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        enqueue_threads_01 = qr_01.create_threads(sess, coord=coord, start=True)
        enqueue_threads_02 = qr_02.create_threads(sess, coord=coord, start=True)
        saver = tf.train.Saver(max_to_keep=None)
        
        if not FLAGS.isresume:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.checkpoint))
            print("Restore complete : %s"%os.path.join(FLAGS.model_dir, FLAGS.checkpoint))
        try:
            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)
            for var, val in zip(tvars, tvars_vals):
                print(var.name)
            
            for i in range(iteration):
                if coord.should_stop():
                    break
                
                A_batch, B_batch = sess.run([dequeue_op_A, dequeue_op_B])
                
                _, _, gen_A_batch_npy, gen_B_batch_npy, lg, lgA, lgB, ld, ls = sess.run([
                    disc_trainer,
                    gen_trainer,
                    gen_BA,
                    gen_AB,
                    loss_gen,
                    loss_total_gen_A,
                    loss_total_gen_B,
                    loss_disc,
                    loss_total_disc_style],
                    feed_dict={A: A_batch, B: B_batch})
                
                now = time.localtime()
                if (i%100 == 0):
                    # 1. Total generator loss
                    # 2. generator_A loss
                    # 3. generator_B loss
                    # 4. Total Discriminator loss
                    # 5. Total Style loss
                    print('%02d-%02d %02d:%02d:%02d '
                            %(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec),'iter ', i, " ", lg, " ", lgA, " ", lgB, " ", ld, " ", ls)
                if (i%5000 == 0 and i == 0):
                    if not FLAGS.isresume:
                        saver.save(sess, "./model/model-%s"%i)
                    else:
                        saver.save(sess, "./model/model-resume-%s"%i)
                if (i%5000 == 0 and i > 1):
                    if not FLAGS.isresume:
                        saver.save(sess, "./model/model-%s"%i, write_meta_graph=False)
                    else:
                        saver.save(sess, "./model/model-resume-%s"%i, write_meta_graph=False)
                    # TODO npy to audio
                    call_gen_audio_subprocess(gen_A_batch_npy, i, "BA")
                    call_gen_audio_subprocess(gen_B_batch_npy, i, "AB")
        except Exception as e:
            # Report exceptions to the coordinator
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(enqueue_threads_01)
            coord.join(enqueue_threads_02)


def inference_input_A_output_B():
    input_data = np.load(os.path.join(FLAGS.input_dir, FLAGS.input_data))
    input_data = np.reshape(input_data, (1, np.shape(input_data)[0], np.shape(input_data)[1], 1))
    
    with tf.Session() as sess:
        # load meta graph
        imported_meta = tf.train.import_meta_graph(FLAGS.model_dir + "model-0.meta")
        # restore weights
        imported_meta.restore(sess, FLAGS.model_dir + FLAGS.checkpoint)
        
        # access network nodes
        graph = tf.get_default_graph()
        input_A = graph.get_tensor_by_name("input/A_data:0")
        output_B = tf.get_collection('generated_B')[0]
        
        result = sess.run(output_B, feed_dict={input_A : input_data})
        result = np.reshape(result, newshape=(config.HEIGHT, config.WIDTH, channel))
        np.save('%s/%s'%(INF_RESULT_NPY_OUTPUT_IS_B, FLAGS.input_data), result)
        subprocess.Popen("python gen_audio.py --input_dir=%s --input=%s --output_dir=%s --output=%s.mp3"
            %(INF_RESULT_NPY_OUTPUT_IS_B, FLAGS.input_data, INF_RESULT_MP3_OUTPUT_IS_B, FLAGS.input_data), shell=True)


def inference_input_B_output_A():
    input_data = np.load(os.path.join(FLAGS.input_dir, FLAGS.input_data))
    input_data = np.reshape(input_data, (1, np.shape(input_data)[0], np.shape(input_data)[1], 1))
    
    with tf.Session() as sess:
        # load meta graph
        imported_meta = tf.train.import_meta_graph(FLAGS.model_dir + "model-0.meta")
        # restore weights
        imported_meta.restore(sess, FLAGS.model_dir + FLAGS.checkpoint)
        
        # access network nodes
        graph = tf.get_default_graph()
        input_B = graph.get_tensor_by_name("input/B_data:0")
        output_A = tf.get_collection('generated_A')[0]
        
        result = sess.run(output_A, feed_dict={input_B : input_data})
        result = np.reshape(result, newshape=(config.HEIGHT, config.WIDTH, channel))
        np.save('%s/%s'%(INF_RESULT_NPY_OUTPUT_IS_A, FLAGS.input_data), result)
        subprocess.Popen("python gen_audio.py --input_dir=%s --input=%s --output_dir=%s --output=%s.mp3"
            %(INF_RESULT_NPY_OUTPUT_IS_A, FLAGS.input_data, INF_RESULT_MP3_OUTPUT_IS_A, FLAGS.input_data), shell=True)


if __name__ == "__main__":
    if(FLAGS.istarget):
        if(FLAGS.is_gen_B):
            inference_input_A_output_B()
        if(FLAGS.is_gen_A):
            inference_input_B_output_A()
    elif(FLAGS.istest):
        print("It will be implemented")
        exit()
    else:
        train()
