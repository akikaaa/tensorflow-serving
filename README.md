This tutorial introduces two basic methods to deploy tensorflow model. Choose appropriate one to deploy your model based on how you train your model.  

	Mode 1: build graph and use session.run;  
	Mode 2: Estimator.  

==========================================================================================

The file distribution of this tutorial is as follow：  
	
	tensorflow-serving/  
		build_server.sh  
		client.py  
		client_estimator.py  
		train_and_export.py   
		train_and_export_estimator.py  
		export/  
			SOMEFOLDER/  

===========================================================================================

	Prerequisites：  
		python 3.X  
		tensorflow 1.10.0  
		tensorflow_serving  
		grpc  
		docker  
    
==========================================================================================

The process of deploying tensorflow model can be divided to: train, export, build and start server, run client.

#### Train:  
Mode1:  build graph + tf.Session.run  
Mode2:  create model_fn, generate estimator, create input_fn, then use estimator.train to train model.
  
#### Export:  
Mode1: create a SavedModelBuilder, then create a Signature which contain input/output format infomation and model type (predict, clissification, etc), finally add graph, variables and Signature to SavedModelBuilder and save. When save, the key of signature_def_map will be used in client. It determines the usage of model when it's called.  
Mode2: first create a feature_spec containing input information, then apply build_parsing_serving_input_receiver_fn together with feature_spec to create input processing function serving_input_receiver_fn. Finally export model.  
  
#### Build and start server:  
saved_model_cli could be used to inspect exported model infomation.  
In both modes, the process of creating server is basically the same. First pull serving image, then create container based on image and exported model. However, due to different methods of calling servce in client side, these two modes needs different port. A suggestion is to use 8500 in mode 1 and 8501 in mode 2. An unsupported port could cause Exception.  
  
#### Run client:  
Mode1: create tensorflow stub and request, define model params in request.model_spec. model_spec.name is MODEL_NAME in building server and model_spec.signature_name is the key of signature_def_map mentioned in Export. Finally import data and send request. Take care of data type here.  
Mode2: set SERVER_URL, like "http://locahost:PORT/v1/models/MODEL_NAME:KEY", where PORT and MODEL_NAME are from building server section, KEY is one of the keys in signature_def_map. Convert data to serialized Examples and send request.  

=============================================================================================

#### Command (take mode 1 as an example):  

	python3 train_and_export.py  

	sudo ./build_server.sh  

	python3 client.py  
