import requests
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
from grpc.beta import implementations
import json
import grpc
import tensorflow as tf
SERVER_URL = 'http://localhost:8500/v1/models/demo:predict'
#
# predict_request = '{"instances":[{"x": [[0.0]]}]}'
# predict_request = json.dumps({'instance': [{'x': [[0.0]]}]})
# response = requests.post(SERVER_URL, data=predict_request)
# print(response.text)
# response.raise_for_status()
# print(response)
server = 'localhost:8500'
# host, port = server.split(':')
# channel = implementations.insecure_channel(host, int(port))
channel = grpc.insecure_channel(server)
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'demo'
request.model_spec.signature_name = "predict"
request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(np.array([[0.0]]).astype(np.float32), shape=[1, 1]))
# request.outputs['y'].CopyFrom(tf.contrib.util.make_tensor_proto(np.array([[0.0]]), shape=[1, 1]))
result_future = stub.Predict(request, 30.)
print(result_future.outputs['y'].float_val)
