import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

inputf = pd.read_csv("assignment6/train_X.csv").values
target = pd.read_csv("assignment6/train_Y.csv").values

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(inputf.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(target.shape[1])
    ])
    return model

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #default learning rate
              loss='mean_absolute_error', 
              metrics=['mae'])

history = model.fit(inputf, target, 
                    validation_split=0.01, 
                    epochs=1000, batch_size=5000, 
                    verbose=1)

plt.clf()
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Loss vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig("Train_vs_Validation_Loss.png")
print("\nSaved Train_vs_Validation_Loss.png to device")
plt.close()

plt.clf()
plt.plot(1 - np.array(history.history['mae']), label='Training Accuracy')
plt.plot(1 - np.array(history.history['val_mae']), label='Validation Accuracy')
plt.title('Model Accuracy vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("\nTrain_vs_Validation_Accuracy.png")
print("Saved Train_vs_Validation_Accuracy.png to device")
plt.close()

model.save("three_body_model.h5")

final_train_acc = 1 - history.history['mae'][-1]
final_val_acc = 1 - history.history['val_mae'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")


# visualizing trajectories
predictions = model.predict(inputf[2002:2012]) #predicting random samples

plt.clf()
plt.plot(target[2002:2012, 0], target[2002:2012, 1], 'go-', label="True Trajectory", markersize=8)
plt.plot(predictions[:10, 0], predictions[:10, 1], 'bo--', label="Predicted Trajectory", markersize=8)

plt.title("Sample Trajectory")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.savefig("Sample_Trajectory_3.png")
print("\nSaved Sample_Trajectory_3.png to device")
plt.show()

