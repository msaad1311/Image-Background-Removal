import onnx

from onnx_tf.backend import prepare

onnx_model = onnx.load("segmentation.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("segs.pb")  # export the model