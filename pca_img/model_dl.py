from pca_img.import_lib import *

def sequential_model_fit(batch_size, num_classes, epochs, metric, x_train, y_train, x_test, y_test, input_shape):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=[metric])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(x_test, y_test))

