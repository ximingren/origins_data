from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
import numpy as np
from pyspark.shell import sc

data=[
    LabeledPoint(0.0,[0.0]),
    LabeledPoint(1.0,[1.0]),
    LabeledPoint(3.0,[2.0]),
    LabeledPoint(2.0,[3.0])
]
lrm=LinearRegressionWithSGD.train(sc.parallelize(data),iterations=10,initialWeights=np.array([1.0]))
print(lrm.predict(np.array([0.0])))