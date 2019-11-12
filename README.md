# Reliability analysis

The Python package [calibration](https://github.com/uu-sml/calibration) provides different
tools for the evaluation of model calibration in classification.

## Installation

You can install the package by running
```shell
pip install git+https://github.com/uu-sml/calibration.git
```

## Usage

All tools for evaluating model calibration are based on the predictions
of your model on a labelled validation data set. Hence prior to any
analysis you have to load a validation data set and compute the
predicted class probabilities of your model on it.

```python
# `onehot_targets` should be an array of the one-hot encoded labels of
# shape (N, C) where N is the number of data points and C the number of classes
inputs, onehot_targets = load_validation_data()

# `predictions` should be an array of the predicted class probabilities of shape
# (N, C) where N is the number of data points and C the number of classes
predictions = model(inputs)
```

You can estimate the expected calibration error (ECE) of your model
with respect to the total variation distance and a binning scheme
with 10 bins of uniform size along each dimension from the
validation data by running:

```python
import calibration.stats as stats

ece = stats.ece(predictions, onehot_targets)
```

Similarly, you can estimate the mean and the standard deviation of
the ECE estimates under the assumption that the model is calibrated:

```python
consistency_ece_mean, consistency_ece_std = stats.consistency_ece(predictions)
```

Alternatively, the bins can be determined from the validation data to achieve
a more even distribution of predictions the bins.

```python
import calibration.binning as binning

ece_datadependent_binning = stats.ece(predictions, onehot_targets, binning=binning.DataDependentBinning())
```

It is also possible to only investigate calibration of certain
aspects of your model by using so-called calibration lenses.
For instance, you can estimate the expected calibration error
using the most confident predictions only.

``` python
import calibration.lenses as lenses

ece_max = stats.ece(*lenses.maximum_lens(predictions, onehot_targets))
```

If you want to know more about additional options and functionalities
of this package, please have a look at the documentation in the source
code.

## Reference

Vaicenavicius J, Widmann D, Andersson C, Lindsten F, Roll J, Schön TB.
**Evaluating model calibration in classification**. PMLR 89:3459-3467, 2019.
[online](http://proceedings.mlr.press/v89/vaicenavicius19a.html).
