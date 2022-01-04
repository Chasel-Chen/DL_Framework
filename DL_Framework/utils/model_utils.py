import tensorflow as tf
from google.protobuf import text_format


def convert_graphdef_to_pbtxt(graphdef_dir, file_name):
    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        with open(graphdef_dir, 'rb') as f:
            _ = tf.import_graph_def(graph_def, name="")
            sess = tf.Session()
            tf.train.write_graph(sess.graph, './', file_name+'.pbtxt')
    return


def convert_pbtxt_to_model(pbtxt_dir, file_name, model_mode):
    assert model_mode in ["pb", "graphdef"], "Unsupport model mode, mode should be pb or graphdef"
    with tf.gfile.FastGFile(pbtxt_dir, 'r') as f:
        graph_def = tf.GraphDef()
        model_content = f.read()
        text_format.Merge((model_content, graph_def))
        tf.train.write_graph(graph_def, './', file_name+'.'+model_mode)
        return
