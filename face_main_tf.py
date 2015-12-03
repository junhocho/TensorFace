#!/usr/bin/env python
#Change CUDA-7.0 to CUDA-7.0-cudnnv2
#source ~/.bash_profile
#echo $CUDA_HOME

# Configuration Part
#CAFFE_PATH = '/home/junho/jeewangue/caffe'
#PYCAFFE_PATH = '/home/junho/jeewangue/caffe/python'
TENSORFACE_PATH = '/home/junho/TensorFace/'
#MODELS_PATH = '/home/junho/jeewangue/DL/models'
DATA_PATH = '/home/pil/Dataset/face'
DB_PATH = '/home/junho/jeewangue/DL/db/'

# Import Libraries, Load VGG_FACE net. Visualize dim of the net

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import Image
import socket
import time
import datetime
import pickle
import glob
import heapq
import sqlite3
import io

os.environ['DISPLAY']=':1'
os.chdir(TENSORFACE_PATH)
print os.path.dirname(os.path.realpath(__file__))
import vggface
from pprint import pprint
import tensorflow as tf

print('import & configure')

### TensorFlow Setup

input_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
network = vggface.VGGFace()
ses = tf.InteractiveSession()
network.load(ses,input_placeholder)

print('TensorFlow vggface setup done')

# given im_path, extract descriptor
def face_discripor_tf(im_path):
    output = network.eval(feed_dict={input_placeholder:vggface.load_image(im_path)})[0]
    return output

####### functions related to DB
# Merge two db
def merge_db(db1, db2):
    return {'im_path':db1['im_path']+db2['im_path'],
            'dscr': np.concatenate([db1['dscr'],db2['dscr']])}

def load_db(name):
    with open(os.path.join(DB_PATH, './face/'+name+'_list'), 'rb') as f:
        im_path = pickle.load(f)
    dscr = np.load(os.path.join(DB_PATH, './face/'+name+'_dscr.npy'))
    return {'im_path':im_path,
            'dscr': dscr}

def save_db(db,name):
    with open(os.path.join(DB_PATH, './face/'+name+'_list'), 'wb') as f:
        pickle.dump(db['im_path'], f)
    np.save(os.path.join(DB_PATH, './face/'+name+'_dscr.npy'), db['dscr'])


# Make lists of im_path
face_path = '/home/pil/Dataset/face/'
db_celeb=load_db('celeb_new_path_fixed')

print len(db_celeb['im_path'])
print db_celeb['dscr'].shape

print "dataload done"

# Define L2dist retrieval function and compare each other
def cal_L2dist(vec1,vec2):
    diff= vec1-vec2
    return np.sum(np.abs(diff) ** 1.2)
#    return (np.sum(diff**2) / vec1.shape / 2.)[0]

# Retrieval closest
def min_L2dist(vec,vectors,num_cand):
    L2dist_with_index=[(cal_L2dist(vec, v),i) for i, v in enumerate(vectors)]
    closest = heapq.nsmallest(num_cand ,L2dist_with_index)
    index_L2=[index for (dist, index) in closest]
    return index_L2

def visualize_top_k(_input, db, k=10,crop=False):
    lst_discriptor = db['dscr']
    lst_im_path = db['im_path']
    print len(lst_im_path)

    if type(_input) is int: #Find from db
        x=min_L2dist(lst_discriptor[_input],lst_discriptor,k)
    elif type(_input) is str: #Given image path
        try:
            im_dscr = face_discripor_tf(_input)
        except Exception:
            print _input[-3:]
            if _input[-3:] == 'jpg':
                _input=_input[:-3]+'png'
            elif _input[-3:] == 'png':
                _input=_input[:-3]+'jpg'
            else:
                raise Exception('unacceptable file format. '+_input)
            im_dscr = face_discripor_tf(face_path+_input)
        #plt.figure()
        im=Image.open(_input)
        #plt.imshow(im)
        x=min_L2dist(im_dscr,lst_discriptor,k)
    close_faces = [lst_im_path[p] for p in x]
    return close_faces
print "function def done"


### TEST
# %matplotlib inline
# test_img = '../../../../home/junho/jeewangue/etc/h2.png'
# imgs = visualize_top_k(test_img, db_friendsceleb, k = 100)
# #urls = map (lambda x : os.path.join('/home/pil/data/' , x) , imgs)
# im_paths = map (lambda x : os.path.join('/home/pil/' , x[3:]), imgs)
# print im_paths
# for im_path in im_paths:
#     plt.figure()
#     im=Image.open(im_path)
#     plt.imshow(im)


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

con = sqlite3.connect(os.path.join(DB_PATH, 'online.db'), detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("create table if not exists face (im_path text, dscr array)")
cur.execute("select im_path from face")
online_im_path = map(lambda x: x[0], cur.fetchall())
print online_im_path
cur.execute("select dscr from face")
online_dscr = map(lambda x: x[0].reshape(1, 4096), cur.fetchall())
print online_dscr
print type(online_dscr[0])
print db_celeb['dscr'][0].shape
print db_celeb['dscr'][0]

db_celeb['im_path'] += online_im_path
db_celeb['dscr'] = np.concatenate([db_celeb['dscr'], np.concatenate(online_dscr)])

########################## input #########################
HOST = 'localhost'
PORT = 12346

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 12346)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()

    try:
        print >>sys.stderr, 'connection from', client_address

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024)
            print >>sys.stderr, 'received "%s"' % data
            if data:
                try:
                    imgs = visualize_top_k(data, db_celeb, k = 10)
                    originalImagePath = os.path.join('online', os.path.basename(data))
                    dscr = face_discripor_tf(face_path + originalImagePath)
                    dscr = dscr.reshape(1, 4096)
                    db_celeb['im_path'] += [originalImagePath]
                    db_celeb['dscr'] = np.concatenate([db_celeb['dscr'], dscr])
                    print db_celeb['dscr'].shape
                    cur.execute("""insert into face(im_path, dscr) values(?, ?)""", (originalImagePath, dscr))
                    con.commit()
                except Exception:
                    imgs = []
                result = ' '.join(imgs)
                result += " END"
                print(result)
                connection.sendall(result)
            else:
                print >>sys.stderr, 'no more data from', client_address
                break
    finally:
        # Clean up the connection
        connection.close()
