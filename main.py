import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sb

digits = load_digits()
dir(digits)
# in df.data each row representing a digit on a 8x8 table (64 cols)
# access row 234
digits.data[234]

# in df.image each digit is encoded as an image
# print first 10 images
plt.gray()
for i in range(13):
    plt.matshow(digits.images[i])

# get digits in i=0 to 4, which is equal to 0 to 4
digits.target[0:5]

# split train and test sets with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=.2)

# fit model
model = LogisticRegression()
model.fit(X_train, y_train)
# check accuracy
model.score(X_test, y_test)

# predict digits[67]
plt.matshow(digits.images[67])
digits.target[67]
model.predict([digits.data[67]])
# predict digits[0 to 9]
model.predict(digits.data[0:10])

# confusion matrix between actual and predicted digits (10x10)
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
# plot heatmap of confusion matrix
plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


