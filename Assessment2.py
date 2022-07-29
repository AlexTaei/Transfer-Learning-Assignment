import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


batch_size = 32
img_height = 256
img_width = 256
img_shape = (img_height, img_width) + (3,)

n_epochs = 20

out_dir = './results'

os.makedirs(out_dir, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(out_dir, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# --------------------------------
# Task 1
def task_1():
    data_dir = './small_flower_dataset'
    return data_dir
# --------------------------------
# Task 2
def task_2():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = img_shape,
        include_top =False,
        weights = 'imagenet'
    )

    base_model.trainable = False
    return base_model

# --------------------------------
# Task 3
def task_3():
    inputs = keras.Input(shape=img_shape)
    x = inputs
    scale_layer = keras.layers.Rescaling(scale =1/127.5, offset = -1)
    x = scale_layer(x)
    x = base_model(x, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(5, activation = 'softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

# --------------------------------
# Task 4
def task_4():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels= 'inferred',
        validation_split = 0.2,
        subset = 'training',
        seed=1234,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels= 'inferred',
        validation_split = 0.2,
        subset = 'validation',
        seed=1234,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    test_ds = val_ds.take(4)
    val_ds = val_ds.skip(4)

    return train_ds, val_ds, test_ds

# --------------------------------
# Task 5
def task_5():
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

    model.compile(
        optimizer = opt,
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs = n_epochs,
        validation_data = val_ds
    )
    return model, history

# --------------------------------
# Task 6
def task_6():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,2.0])
    plt.yticks(np.arange(0,2,0.1))
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    save_fig(f"Initial training")

# --------------------------------
# Task 7
def task_7():
    old_weights = model.get_weights()
    learning_rates = [0.1, 0.01, 0.001]

    for rate in learning_rates:
        model.set_weights(old_weights)
        opt = tf.keras.optimizers.SGD(learning_rate=rate, momentum=0.0, nesterov=False)

        model.compile(
            optimizer= opt,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        # plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        save_fig(f"{rate}_test")

# --------------------------------
# Task 8
def task_8():
    old_weights = model.get_weights()
    momentum_tests = [0.25, 0.5, 0.75]
    lr = 0.01
    for momentum in momentum_tests:
        model.set_weights(old_weights)
        opt = tf.keras.optimizers.SGD(learning_rate = lr, momentum = momentum, nesterov=False)

        model.compile(
            optimizer= opt,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        # plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        save_fig(f"lr_{lr}_momentum_{momentum}_test2")
        # plt.show()

def PredictWithBestParameters():
    momentum = 0.5
    lr = 0.01
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=False)

    model.compile(
        optimizer = opt,
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    model.fit(
        train_ds,
        epochs = n_epochs,
        validation_data = val_ds
    )

    image_batch, label_batch = test_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch)
    predictions = np.argmax(predictions, axis =1)
    class_names = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")
    plt.show()



if __name__ == "__main__":
    data_dir = task_1()
    base_model = task_2()
    model = task_3()
    train_ds, val_ds, test_ds = task_4()

    # Run task 5 and task 6 together to train with default given values and plot
    # model, history = task_5()
    # task_6()

    # Run task 7 by itself to test different magnitudes of learning rate
    # task_7()

    # Run task 8 by itself to test different magnitudes of momentum with the best learning rate
    # task_8()

    # Run this by it self to predict with the best learning rate and momentum  
    PredictWithBestParameters()

    pass








