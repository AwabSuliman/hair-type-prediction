from imageProcessing import resize_images
from sklearn.model_selection import train_test_split

Labels = ["curly", "dreadlocks", "kinky", "straight", "wavy"]
filePath = "/Users/awab/Desktop/hair/data"
X, Y = resize_images(filePath, Labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1 )

print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", Y_train.shape)
print("Y_test Shape:", Y_test.shape)

