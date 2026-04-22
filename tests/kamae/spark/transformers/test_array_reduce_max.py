import numpy as np
import pytest
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType
from kamae.spark.transformers import ArrayReduceMaxTransformer


class TestArrayReduceMaxTransformer:
    @pytest.fixture(scope="class")
    def input_df(self, spark_session):
        return spark_session.createDataFrame(
            [
                ([3.0, 1.0, 2.0],),
                ([0.0, 5.0, 4.0],),
                ([-3.0, -1.0, -2.0],),
            ],
            ["values"],
        )

    def test_returns_maximum_of_each_row(self, input_df):
        transformer = ArrayReduceMaxTransformer(inputCol="values", outputCol="result")
        result = transformer.transform(input_df).select("result").collect()

        assert [row.result for row in result] == pytest.approx([3.0, 5.0, -1.0])

    def test_default_value_for_empty_array(self, spark_session):

        schema = StructType([
            StructField("values", ArrayType(FloatType()), nullable=True)
        ])

        df = spark_session.createDataFrame(
            [([],)],
            schema=schema
        )

        transformer = ArrayReduceMaxTransformer(
            inputCol="values", outputCol="result", defaultValue=-99.0
        )
        result = transformer.transform(df).select("result").collect()

        assert result[0].result == pytest.approx(-99.0)

    def test_spark_keras_parity(self, input_df):
        import tensorflow as tf

        transformer = ArrayReduceMaxTransformer(inputCol="values", outputCol="result")
        spark_values = (
            transformer.transform(input_df)
            .select("result")
            .rdd.map(lambda r: r[0])
            .collect()
        )

        inputs = tf.constant([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0], [-3.0, -1.0, -2.0]])
        keras_values = transformer.get_tf_layer()(inputs).numpy().tolist()

        np.testing.assert_array_almost_equal(spark_values, keras_values)