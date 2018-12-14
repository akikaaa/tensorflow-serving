HOME=./
# saved_model_cli show --dir $HOME/export/1/ --all

# docker pull tensorflow/serving


# create container
# change port to 8501:8501 for estimator version
nvidia-docker run -p 8500:8500 --name tfserving_demo \
--mount type=bind,source=$HOME/export,target=/models/demo \
-e MODEL_NAME=demo -t tensorflow/serving &

# start container when it's already created.
# nvidia-docker start -i tfserving_demo
