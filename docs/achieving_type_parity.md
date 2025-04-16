# Type Parity in Kamae

By default, Spark and Tensorflow have very different behaviours when it comes to the data types of the outputs of their transforms.

For example, computing `pyspark.sql.functions.log(x)` will always return a `DoubleType` column, even if `x` is an `IntegerType` or `FloatType` column.
On the other hand, Tensorflow often attempts to maintain input and output datatypes, and so if you pass a `tf.float32` tensor into a `tf.math.log` operation, you will get a `tf.float32` tensor out.

This inconsistency can cause issues if you plan to write the output of your Kamae pipeline to TFRecords for later training stages.
In these cases you could be left with a mismatch between the expected and actual data types of your features.

## How to achieve type parity

Kamae provides `inputDtype` and `outputDtype` parameters to every Spark transformer/estimator that will cast 
the inputs and outputs to the specified data type. This is mimicked on the Tensorflow side, so that casting is done in a consistent manner.
These can be used to ensure:

1. The input datatype is a compatible datatype for the transformer/estimator.
2. For example the `LogTransformer` can only operate on floating types, but by specifying `inputDtype="float"` you could pass in an `IntegerType` column and it would be cast to a `FloatType` column before the log operation is applied.
3. The output datatype is a compatible datatype for the next stage of the pipeline.


## Pitfalls

There is one special case where even providing `inputDtype` and `outputDtype` may still not achieve type parity.
In the case where you intend to return a `string` from a numerical operation, setting `outputDtype="string"` can have unexpected results.

- Casting a double to string in Tensorflow currently only preserves 6 significant figures, which can lead to loss of precision. In Spark all decimal places are preserved.
- Casting an integer to a string is different to casting a float (of the same integer value) to a string.
  - For example, casting `1` to a string will return `"1"`. Casting `1.0` to a string will return `"1.0"`.

Some operations in Spark always return DoubleType, whereas in Tensorflow they can return integers if the inputs are integers.
If you then set the outputDtype to "string" you will get different results from the two frameworks.

In these cases, it is recommended to set `outputDtype` to some intermediary numerical type (e.g. `float`) and then cast to string in a separate step.
This separate step can be done using the `IdentityTransformer` with `outputDtype="string"`.

Lastly, it is worth noting that setting `inputDtype` and `outputDtype` will add some small overhead/latency to your resulting keras model, since we will perform a casting operation on the inputs. However this overhead should be minimal in comparison to the rest of the operations in your pipeline. 

## Rules

Consolidating all the above into a set of rules to achieve type parity:

1. Always set `outputDtype` on every transformer/estimator.
2. If you require a string output from a numerical operation, set `outputDtype` to an intermediary numerical type and then cast to string in a separate step (using the `IdentityTransformer` with `outputDtype="string"`).
