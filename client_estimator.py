import tensorflow as tf
import base64
import requests


def convert_inputs_to_serialized_examples(inputs):
    """

    :param inputs: {'x': list}
    :return: a list of serialized tf.train.Example
    """
    xs = inputs['x']
    serialized = []

    def create_float_feature(value):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return f

    for x in xs:
        features = {'x': create_float_feature(x)}
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        serialized_example = tf_example.SerializeToString()
        serialized.append(serialized_example)
        # features_list.append(features)
    return serialized


def main():
    server_url = 'http://localhost:8501/v1/models/demo:predict'
    inputs = {'x': [0.0]}
    serialized = convert_inputs_to_serialized_examples(inputs=inputs)
    predict_request = '{"instances": ['
    for i in range(len(serialized)):
        if i == 0:
            cur_string = '{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
        else:
            cur_string = ',{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
        predict_request += cur_string
    predict_request += ']}'
    response = requests.post(server_url, data=predict_request)
    # print(response.text)
    response.raise_for_status()
    response = response.json()
    predictions = response['predictions']
    print(predictions)


if __name__ == "__main__":
    main()