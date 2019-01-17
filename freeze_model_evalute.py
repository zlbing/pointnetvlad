
import argparse
import tensorflow as tf

from loading_pointclouds import *

DATABASE_FILE= 'generating_queries/kaicheng_evaluation_database.pickle'
DATABASE_SETS= get_sets_dict(DATABASE_FILE)
NUM_POINTS = 4096
BATCH_NUM_QUERIES = 1
POSITIVES_PER_QUERY = 0
NEGATIVES_PER_QUERY = 0

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    print("load graph success")
    return graph

def get_latent_vectors(graph, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))
    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY)
    q_output = []
    file_names=[]
    file_indices=train_file_idxs[0*batch_num:(0+1)*(batch_num)]
    for index in file_indices:
        file_names.append(dict_to_process[index]["query"])
    queries=load_pc_files(file_names)

    q1=queries[0:BATCH_NUM_QUERIES]
    q1=np.expand_dims(q1,axis=1)

    q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
    q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

    q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
    q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))

    query = graph.get_tensor_by_name("Placeholder:0")
    positives = graph.get_tensor_by_name("Placeholder_1:0")
    negatives = graph.get_tensor_by_name("Placeholder_2:0")
    is_training_pl = graph.get_tensor_by_name("Placeholder_4:0")
    y = graph.get_tensor_by_name('query_triplets/Reshape_5:0')
    feed_dict={query:q1,
               positives:q2,
               negatives:q3,
               is_training_pl: is_training}

    with tf.Session(graph=graph) as sess:
        q_output = sess.run(y, feed_dict)
        print("q_output.shape=",q_output.shape)
        #print(y_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="models/refine/refine_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    for op in graph.get_operations():
        print(op.name,op.values())

    get_latent_vectors(graph, DATABASE_SETS[0])