# LOGS

Sub File     | LB score | Description
-------------|----------|---------------------------------------------
Sub1         | 0.7965   | ResNet transfer learned
Sub2         | 0.81295  | ResNet fine-tuned
Ensemble1    | 0.80720  | AM (Sub1, Sub2)
Ensemble2    | 0.80984  | GM (Sub1, Sub2)
Sub3         | 0.0      | DenseNet unfreezed - 0
Sub4         | 0.0      | DenseNet unfreezed - 5
Sub5         | 0.85464  | DenseNet unfreezed - 0 (SOFTMAX)
Sub6         | 0.8431   | ResNet (non-augmented) (SOFTMAX)
Ensemble3    | 0.8762   | AM (Sub5, Sub6)
Ensemble4    | 0.8910   | GM (Sub5, Sub6)
Sub7         | 0.8740   | DenseNet unfreezed - 20 (best single model so far)
Ensemble5    | 0.89815  | GM (Sub6, Sub7)
Ensemble6    | 0.88659  | AM (Sub6, Sub7)
Sub8         | 0.       | ResNet unfreezed - 


Notes:

1. Softmax is better.
2. GM is better.
