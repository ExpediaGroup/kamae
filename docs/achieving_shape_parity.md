# Shape Parity in Kamae

Traditionally, the Spark DataFrame API had poor support for nested data structures, such as arrays and maps. Nowadays, Spark has improved its support for nested data structures, but it is still not as flexible as Tensorflow.

In order to improve the compatibility between Spark and Tensorflow, Kamae provides a set of utils for transforming nested arrays that are used in all Spark-side transformers.

This allows the user to maintain shape parity between the two frameworks.

## How to achieve shape parity

You can now have features of any level of nesting in your schema, provided they are only arrays. Not maps or structs. 

All transformers and estimators now natively support nested inputs, and can operate elementwise or across the full array in accordance with how 
Tensorflow would operate on the same data.

## Restrictions

- The nested arrays must be homogenous. That is, all elements of the array must be of the same type.
- In the case of multiple input transformers, the nested arrays must be of the same size, or be a scalar. Scalars are broadcast to the size of the other array(s).
- The operation is performed either elementwise (in the case of a simple transform such as `log(x + 1)`) or across the innermost array (`axis=-1` in Tensorflow).
