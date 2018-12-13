#saved_model_cli show --dir PATH/TO/SAVED/MODEL --all

#docker pull tensorflow/serving

nvidia-docker run -p 8500:8500 --name tfserving_demo \
--mount type=bind,source=/home/ycl/personal/github/serving-demo/export,target=/models/demo \
-e MODEL_NAME=demo -t tensorflow/serving &
