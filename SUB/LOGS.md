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
Sub8         | 0.81328  | ResNet unfreezed - 20
Ensemble7    | 0.88022  | GM (Sub7, Sub8)
Sub9         | 0.78396  | Xception unfreezed - 4 (Need tuning)
Ensemble8    | 0.85852  | GM (Sub8, Sub9)
Ensemble9    | 0.88322  | GM (Sub7, SUb9)
Ensemble10   | 0.88861  | GM (Sub7, Sub8, Sub9)
Ensemble11   | 0.89845  | GM (Sub7, Sub6, Sub9)
Ensemble12   | 0.90150  | GM (Sub7 * 2, Sub6, Sub9)


Notes:

1. Softmax is better.
2. GM is better.
