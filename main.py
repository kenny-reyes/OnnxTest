from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import __max_supported_opset__, to_onnx
import pickle

print("Last supported opset:", __max_supported_opset__)

# Train a model.
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

# clr = RandomForestClassifier()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

filename = 'main.pkl'
pickle.dump(model, open(filename, 'wb'))

# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)
#onx = to_onnx(clr, X_train[:1])
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

prediction = model.predict(X_test)

loaded_model = pickle.load(open(filename, 'rb'))
pre2 = loaded_model.predict(X_test)

# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy

sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

print(pred_onx)