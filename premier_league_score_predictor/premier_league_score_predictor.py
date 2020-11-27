import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, losses, metrics
from keras.layers import Dense, Activation
from keras import backend as K
from keras.utils import to_categorical

# Loading in the data: 460 rows split into 370 training sample and 90 test sample
train_data = np.loadtxt('train_data.csv', delimiter=',')
train_targets = np.loadtxt('train_targets.csv', delimiter=',')
test_data = np.loadtxt('test_data.csv', delimiter=',')
test_targets = np.loadtxt('test_targets.csv', delimiter=',')
prediction_data = np.loadtxt('prediction_data.csv', delimiter=',')

## Checking the data
#print('train_data looks like this:\n ',train_data, '\n')

# Normalising the Data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data/= std

test_data -= mean
test_data /= std

# Checking the data 
print('train_data after normalisation: ',train_data)

# Defining the model
def build_model():
    output_dim = nb_classes = 10 
    model = models.Sequential() 
    model.add(Dense(24, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(Dense(44, activation='relu'))
    model.add(Dense(24, activation='sigmoid',))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy']) 
    return model

# Validating data using k-folds (remember has to be a factor of len(train_data)!
k = 5
num_val_samples = len(train_data)//k
num_epochs = 9
all_scores=[]

for i in range(k):
    print('Processing fold #',i)
    # Prepare validation data: data from partition #k
    val_data = train_data[int(i * num_val_samples): int((i+1) * num_val_samples)]
    val_targets = train_targets[int(i * num_val_samples): int((i+1) * num_val_samples)]

    # Prepare training data: data from all other partiations
    partial_train_data = np.concatenate([train_data[:int(i * num_val_samples)], train_data[int((i+1) * num_val_samples):]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:int(i * num_val_samples)], train_targets[int((i+1) * num_val_samples):]], axis=0)

# Compiling model
model = build_model()
history = model.fit(partial_train_data, partial_train_targets,
          batch_size=1, epochs=num_epochs, verbose=0,
          validation_data=(val_data, val_targets
          ))

# Evaluating the model on the validation data
val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
all_scores.append(val_mae)



# Test data
results = model.evaluate(test_data, test_targets, verbose=0)
print('Test score:', results[0]) 
print('Test accuracy:', results[1])

# Plotting
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Normalizing predictions
prediction_data -= mean
prediction_data /= std

# Evaluating the predictions data
pred = model.predict(prediction_data)


print('\npredictions:\n', pred)