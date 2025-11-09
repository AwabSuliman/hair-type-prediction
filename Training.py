from imageProcessing import resize_images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import joblib

Labels = ["curly", "dreadlocks", "kinky", "straight", "wavy"]
filePath = "/Users/awab/Desktop/hair/data"
X, Y = resize_images(filePath, Labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=67)

#data encoding
encoder = LabelEncoder()
Ytrain_encoded = encoder.fit_transform(Y_train)
Ytest_encoded = encoder.transform(Y_test)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# LinearSVCmodel = LinearSVC(max_iter=50000)
# LinearSVCmodel.fit(X_train_scaled, Ytrain_encoded) 

# KNeighborsClassifiermodel = KNeighborsClassifier(n_neighbors=5)
# KNeighborsClassifiermodel.fit(X_train_scaled, Ytrain_encoded) 

# SVCmodel = SVC(max_iter=50000)
# SVC.fit(X_train_scaled, Ytrain_encoded) 

# LogisticRegressionmodel = LogisticRegression(max_iter=50000)
# LogisticRegressionmodel.fit(X_train_scaled, Ytrain_encoded) 


model = SVC(kernel='rbf', max_iter=20000)


model.fit(X_train_scaled, Ytrain_encoded)
y_pred = model.predict(X_test_scaled)
acc = metrics.accuracy_score(Ytest_encoded, y_pred)
print("accuracy is:", acc)

joblib.dump(model, "hair_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")




# print("X_train Shape:",  X_train.shape)
# print("X_test Shape:", X_test.shape)
# print("Y_train Shape:", Y_train.shape)
# print("Y_test Shape:", Y_test.shape)


