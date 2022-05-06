from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
import numpy
from onnxruntime import InferenceSession
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    VotingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train classifiers
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=5)
reg2 = RandomForestRegressor(random_state=1, n_estimators=5)
reg3 = LinearRegression()

ereg = Pipeline(steps=[
    ('voting', VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)])),
])
ereg.fit(X_train, y_train)


onx = to_onnx(ereg, X_train[:1].astype(numpy.float32), target_opset=12)

sess = InferenceSession(onx.SerializeToString())
pred_ort = sess.run(None, {'X': X_test.astype(numpy.float32)})[0]

pred_skl = ereg.predict(X_test.astype(numpy.float32))

print("Onnx Runtime prediction:\n", pred_ort[:5])
print("Sklearn rediction:\n", pred_skl[:5])