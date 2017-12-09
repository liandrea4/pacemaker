import numpy as np
import os
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sklearn.model_selection 		 import train_test_split
from keras.applications.vgg16 		 import VGG16
from keras.preprocessing.image       import ImageDataGenerator
from keras.models                    import Sequential, Model, Input
from keras.layers                    import Dense, Flatten, Dropout
from keras.optimizers                import SGD, RMSprop, Adam
from keras.callbacks                 import EarlyStopping
from keras.utils                     import np_utils
from keras                           import backend as K

# only use GPUID == 1
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run(X, Y, pickle_filename, model_filename, batch_size=32, num_epochs=50): 

	# Split into test and train/validation set
	X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)

	# Split into train and validation
	X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, train_size=0.9, random_state=42, stratify=y_train) 
	print('Train shape: ', X_train.shape, y_train.shape)
	print('Val shape: ', X_val.shape, y_val.shape)
	print('Test shape: ', X_test.shape, np.array(y_test).shape)
    
    

	# Categorize the labels
	num_classes = 2
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_val = np_utils.to_categorical(y_val, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)
	print("y_train, y_val, y_test: ", y_train.shape, y_val.shape, y_test.shape)

	# Create the base pre-trained model
	base_model = VGG16(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

	x = base_model.output
	x = Flatten()(x)
# 	x = Dense(2048, activation='relu')(x)
# 	x = Dropout(.7)(x)
# 	x = Dense(2048, activation='relu')(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=predictions)
	k = 1 # number of end layers to retrain
	layers = base_model.layers[:-k] if k != 0 else base_model.layers
	for layer in layers: 
	    layer.trainable = False
	print(model.summary())

	# Compile model
	opt = SGD(lr=0.0001, momentum=0.9)
	model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

	# Initiate the train, validation and test generators with data augumentation
	train_datagen = ImageDataGenerator(
        rescale = 1./255,
		horizontal_flip = True,
		vertical_flip = True, 
		rotation_range=90,
		zoom_range=0.3,
		fill_mode='nearest'
    )
	train_datagen.fit(X_train)
	generator = train_datagen.flow(
        X_train, 
		y_train, 
		batch_size=batch_size, 
		save_to_dir='/enc_data/eddata/pacemaker/augmented/train/',
		save_format='png'
    )

	val_datagen = ImageDataGenerator(
        rescale = 1./255,
		horizontal_flip = True,
		vertical_flip = True, 
		rotation_range=90,
		zoom_range=0.3,
		fill_mode='nearest',
    )
	val_datagen.fit(X_val)
	val_generator = val_datagen.flow(
		X_val, 
		y_val, 
		batch_size=batch_size, 
		save_to_dir='/enc_data/eddata/pacemaker/augmented/val/',
		save_format='png'
	)

	test_datagen = ImageDataGenerator(
		rescale = 1./255,
		horizontal_flip = True,
		vertical_flip = True, 
		rotation_range=90,
		zoom_range=0.3,
		fill_mode='nearest',
	)
	test_datagen.fit(X_test)
	test_generator = test_datagen.flow(
		X_test, 
		y_test, 
		batch_size=batch_size, 
		save_to_dir='/enc_data/eddata/pacemaker/augmented/test/',
		save_format='png'
	)

	# Train the model, auto terminating when val_acc stops increasing after 10 epochs.
	# callback = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=2, mode='max') 
    # , callbacks=[callback],
	hist = model.fit_generator(val_generator, steps_per_epoch=len(X_val) / batch_size , epochs=num_epochs, verbose=1, validation_data=val_generator, validation_steps=len(X_val)/batch_size)

	# Save accuracy / loss during training to pickle file so we can plot later
	# pickle.dump(hist.history, open(pickle_filename, 'wb'))

	# Evalulate model
	test_loss, accuracy = model.evaluate_generator(val_generator, X_val.shape[0])
	print('Test loss: ', test_loss, ' Accuracy: ', accuracy)

	# Save model
	# model.save(model_filename)
