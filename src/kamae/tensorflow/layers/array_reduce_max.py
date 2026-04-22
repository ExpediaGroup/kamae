from typing import Any, Dict, Iterable, List, Optional

import tensorflow as tf

import kamae
from kamae.tensorflow.typing import Tensor
from kamae.tensorflow.utils import enforce_single_tensor_input

from .base import BaseLayer


@tf.keras.utils.register_keras_serializable(package=kamae.__name__)
class ArrayReduceMaxLayer(BaseLayer):
    """
    Reduces the last dimension of a tensor by taking the maximum.

    Input:  (..., N)
    Output: (...)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_dtype: Optional[str] = None,
        output_dtype: Optional[str] = None,
        default_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, input_dtype=input_dtype, output_dtype=output_dtype, **kwargs
        )
        self.default_value = default_value

    @property
    def compatible_dtypes(self) -> Optional[List[tf.dtypes.DType]]:
        return [tf.bfloat16, tf.float16, tf.float32, tf.float64]

    @enforce_single_tensor_input
    def _call(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        result = tf.reduce_max(inputs, axis=-1)
        return tf.where(
            tf.math.is_nan(result),
            tf.constant(self.default_value, dtype=result.dtype),
            result,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"default_value": self.default_value})
        return config