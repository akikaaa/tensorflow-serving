from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import grpc
import tensorflow as tf
server = 'localhost:8500'
channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'demo'
request.model_spec.signature_name = "predict"
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(np.array([[0.0]]).astype(np.float32), shape=[1, 1]))
result_future = stub.Predict(request, 30.)
print(result_future.outputs['y'].float_val)
