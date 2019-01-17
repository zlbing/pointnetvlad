import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pointnetvlad_cls import *
from loading_pointclouds import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

from visualize import *

#params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--positives_per_query', type=int, default=0, help='Number of potential positives in each training tuple [default: 4]')
parser.add_argument('--negatives_per_query', type=int, default=0, help='Number of definite negatives in each training tuple [default: 12]')
parser.add_argument('--batch_num_queries', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--dimension', type=int, default=256)
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

#BATCH_SIZE = FLAGS.batch_size
BATCH_NUM_QUERIES = FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY= FLAGS.positives_per_query
NEGATIVES_PER_QUERY= FLAGS.negatives_per_query
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


RESULTS_FOLDER="results/"
if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER)

DATABASE_FILE= 'generating_queries/kaicheng_evaluation_database.pickle'
QUERY_FILE= 'generating_queries/kaicheng_evaluation_query.pickle'

LOG_DIR = 'models/refine'
output_file= RESULTS_FOLDER +'results.txt'
model_file= "model_refine.ckpt"

DATABASE_SETS= get_sets_dict(DATABASE_FILE)
#print(DATABASE_SETS)
QUERY_SETS= get_sets_dict(QUERY_FILE)
global DATABASE_VECTORS
DATABASE_VECTORS=[]

global QUERY_VECTORS
QUERY_VECTORS=[]

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay     

def evaluate():
    global DATABASE_VECTORS
    global QUERY_VECTORS

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("[evaluate]In Graph")
            query= placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives= placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives= placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            eval_queries= placeholder_inputs(EVAL_BATCH_SIZE, 1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print("query=",query)
            print("positives=",positives)
            print("negatives=",negatives)
            print("eval_queries=",eval_queries)
            print("is_training_pl=",is_training_pl)
           

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            with tf.variable_scope("query_triplets") as scope:
                vecs= tf.concat([query, positives, negatives],1)
                print("vecs=",vecs)                
                out_vecs= forward(vecs, is_training_pl, bn_decay=bn_decay)
                print("out_vecs=",out_vecs)
                q_vec, pos_vecs, neg_vecs= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY],1)
                print("q_vec=",q_vec)
                print("pose_vecs=",pos_vecs)
                print("neg_vecs=",neg_vecs)

            saver = tf.train.Saver()
            
        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, os.path.join(LOG_DIR, model_file))

        output_graph = os.path.join(LOG_DIR,"refine_model.pb")
        print("[evaluate] Model restored.")
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        print("%d ops in the input graph.\n\n" % len(input_graph_def.node))
        output_node_names = "query_triplets/Reshape_5"
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session
            tf.get_default_graph().as_graph_def(), # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")  
        )
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph.\n\n" % len(output_graph_def.node))
        #[print(n.name) for n in output_graph_def.node]
        return

        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'is_training_pl': is_training_pl,
               'eval_queries': eval_queries,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs}
        recall= np.zeros(25)
        count=0
        similarity=[]
        one_percent_recall=[]
        print("[evaluate] DATABASE_SETS size=",len(DATABASE_SETS))
        print("[evaluate] QUERY_SETS size=",len(QUERY_SETS))
        for j in range(len(QUERY_SETS)):
            QUERY_VECTORS.append(get_latent_vectors(sess, ops, QUERY_SETS[j]))

        for i in range(len(DATABASE_SETS)):
            DATABASE_VECTORS.append(get_latent_vectors(sess, ops, DATABASE_SETS[i]))
            print("DATABASE_SETS=", len(DATABASE_SETS[i]))

        for j in range(len(QUERY_SETS)):
            vector_output_file = RESULTS_FOLDER +str(j) +"database.txt"
            with open(vector_output_file, "w") as output:
                for k in range(DATABASE_VECTORS[j].shape[0]):
                    output.write(",".join(str(elem) for elem in DATABASE_VECTORS[j][k,:]) + "\n")
            index_file = RESULTS_FOLDER+str(j) + "index.txt"
            with open(index_file, "w") as output:
                for k in range(len(DATABASE_SETS[j])):
                    output.write(str(DATABASE_SETS[j][k]["query"]) + "\n")
            
        print("EEEEEEEEEEEEEEEEE")


        print("[evaluate] DATABASE_VECTORS size=",len(DATABASE_VECTORS))
        print("[evaluate] QUERY_VECTORS size=",len(QUERY_VECTORS))
        all_dataset_all_indices = []
        for m in range(len(QUERY_SETS)):
            for n in range(len(QUERY_SETS)):
                if(m==n):
                    continue
                print("[evaluate] m=",m,"n=", n)
                pair_recall, pair_similarity, pair_opr, all_indices = get_recall(sess, ops, m, n)
                all_dataset_all_indices.append(all_indices)
                recall+=np.array(pair_recall)
                count+=1
                one_percent_recall.append(pair_opr)
                print("[evaluate] pair_opr =", pair_opr)
                print("[evaluate] pair_similarity=", pair_similarity)
                for x in pair_similarity:
                    similarity.append(x)

        ave_recall=recall/count
        print("[evaluate] ave_recall=",ave_recall)

        #print(similarity)
        average_similarity= np.mean(similarity)
        print("[evaluate] average_similarity=",average_similarity)

        ave_one_percent_recall= np.mean(one_percent_recall)
        print("[evaluate] ave_one_percent_recall=",ave_one_percent_recall)

#        index = 0
#        for m in range(len(QUERY_SETS)):
#            threshold=max(int(round(len(DATABASE_VECTORS[m])/100.0)),1)
#            for n in range(len(QUERY_SETS)):
#                if(m==n):
#                    continue
#                all_indices = all_dataset_all_indices[index]
#                index = index+1
#
#                print("all_indices=",len(all_indices),"query size=",len(QUERY_SETS[n]))
#                for k in range(len(QUERY_SETS[n])):
#                    indices = all_indices[k]
#                    true_neighbors = QUERY_SETS[n][k][m]
#                    if(len(true_neighbors)==0):
#                        continue
#                    print("\n\nm=",m,"n=",n,"k=",k)
#                    if indices[0] in true_neighbors:
#                        print("query=",QUERY_SETS[n][k]["query"])
#                        print("true result database=",DATABASE_SETS[m][indices[0]]["query"])
#                        # fig = plt.figure()
#                        # query_point = load_pc_file(QUERY_SETS[n][k]["query"])
#                        # matplotVisual(query_point, 221,fig, "query"+QUERY_SETS[n][k]["query"])
#                        # loopup_false_database_point = load_pc_file(DATABASE_SETS[m][indices[0]]["query"])
#                        # matplotVisual(loopup_false_database_point, 223,fig,"query true result"+DATABASE_SETS[m][indices[0]]["query"])
#                        # true_database_point = load_pc_file(DATABASE_SETS[m][true_neighbors[0]]["query"])
#                        # matplotVisual(true_database_point, 224, fig, "true result"+DATABASE_SETS[m][true_neighbors[0]]["query"])
#                        # plt.show()
#                    else:
#                        print("query=",QUERY_SETS[n][k]["query"])
#                        print("indices=",indices)
#                        for kk in range(len(indices)):
#                            print("false result database=",DATABASE_SETS[m][indices[kk]]["query"])
#                            if(kk>5):
#                                break
#                        print("true_neighbors=",true_neighbors)
#                        for kk in range(len(true_neighbors)):
#                            print("true result database=",DATABASE_SETS[m][true_neighbors[kk]]["query"])
#
#                        if len(list(set(indices[0:threshold]).intersection(set(true_neighbors))))==0:
#                            ##figure wrong answer
#                            fig = plt.figure()
#                            query_point = load_pc_file(QUERY_SETS[n][k]["query"])
#                            matplotVisual(query_point, 221,fig, "query"+QUERY_SETS[n][k]["query"])
#                            loopup_false_database_point = load_pc_file(DATABASE_SETS[m][indices[0]]["query"])
#                            matplotVisual(loopup_false_database_point, 223,fig,"query false result"+DATABASE_SETS[m][indices[0]]["query"])
#                            true_database_point = load_pc_file(DATABASE_SETS[m][true_neighbors[0]]["query"])
#                            matplotVisual(true_database_point, 224, fig, "true result"+DATABASE_SETS[m][true_neighbors[0]]["query"])
#                            plt.show()

        #filename=RESULTS_FOLDER +'average_recall_oxford_netmax_sg(finetune_conv5).txt'
        with open(output_file, "w") as output:
            output.write("Average Recall @N:\n")
            output.write(str(ave_recall))
            output.write("\n\n")
            output.write("Average Similarity:\n")
            output.write(str(average_similarity))
            output.write("\n\n")
            output.write("Average Top 1% Recall:\n")
            output.write(str(ave_one_percent_recall))
            for k in range(len(DATABASE_SETS)):
                output.write("database["+str(k)+"]="+str(DATABASE_SETS[k][0]["query"])+"\n")


def get_latent_vectors(sess, ops, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    #print(len(train_file_idxs))
    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        file_names=[]
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        #print("[get_latent_vectors] file_indices",file_indices)
        #print("[get_latent_vectors]file_names",file_names)
        queries=load_pc_files(file_names)
        # queries= np.expand_dims(queries,axis=1)
        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)
        #print(q1.shape)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3, ops['is_training_pl']:is_training}
        o1, o2, o3=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs']], feed_dict=feed_dict)

        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))

        out=np.vstack((o1,o2,o3))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):  
        q_output=q_output.reshape(-1,q_output.shape[-1])
    #print(q_output.shape)

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        print("[get_latent_vectors edge case]q_index=",q_index)
        index=train_file_idxs[q_index]
        queries=load_pc_files([dict_to_process[index]["query"]])
        queries= np.expand_dims(queries,axis=1)
        #print(query.shape)
        #exit()
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
        #print(q.shape)
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        #print(output.shape)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    #q_output=np.array(q_output)
    #q_output=q_output.reshape(-1,q_output.shape[-1])
    print("[get_latent_vectors] results shape=",q_output.shape)
    return q_output

def get_recall(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors=25
    recall=[0]*num_neighbors

    top1_similarity_score=[]
    one_percent_retrieved=0
    threshold=max(int(round(len(database_output)/100.0)),1)
    all_indices = []
    num_evaluated=0
    for i in range(len(queries_output)):
        true_neighbors= QUERY_SETS[n][i][m]
        if(len(true_neighbors)==0):
            all_indices.append([])
            continue
        num_evaluated+=1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=num_neighbors)
        all_indices.append(indices[0])
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                #print("j=",j,"distance=",distances[0][j])
                if(j==0):
                    similarity= np.dot(queries_output[i],database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j]+=1
                break
                
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))>0:
            one_percent_retrieved+=1

    one_percent_recall=(one_percent_retrieved/float(num_evaluated))*100
    recall=(np.cumsum(recall)/float(num_evaluated))*100
    print("[get_recall] recall=",recall)
    print("[get_recall] mean=",np.mean(top1_similarity_score))
    print("[get_recall] one_percent_recall=",one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall, all_indices

def get_similarity(sess, ops, m, n):
    global DATABASE_VECTORS
    global QUERY_VECTORS

    database_output= DATABASE_VECTORS[m]
    queries_output= QUERY_VECTORS[n]

    threshold= len(queries_output)
    print("[get_similarity] queries_output shape=",queries_output.shape)
    database_nbrs = KDTree(database_output)

    similarity=[]
    for i in range(len(queries_output)):
        distances, indices = database_nbrs.query(np.array([queries_output[i]]),k=1)
        for j in range(len(indices[0])):
            q_sim= np.dot(q_output[i], database_output[indices[0][j]])
            similarity.append(q_sim)
    average_similarity=np.mean(similarity)
    print("[get_similarity] average_similarity=",average_similarity)
    return average_similarity 


if __name__ == "__main__":
    evaluate()
