# Copyright [2024] Expedia, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import requests


def predict_rest(json_data, url):
    json_response = requests.post(url, data=json_data)
    print("Response: ", json_response.text)
    predictions = json.loads(json_response.text)["predictions"]
    return predictions


if __name__ == "__main__":
    # Change these if the model you're testing has different inputs
    inputs = [
        {
            "col1": [[[1], [4], [7]]],
            "col2": [[[2], [5], [8]]],
            "col3": [[[3], [6], [9]]],
            "col4": [[["a"], ["b"], ["c"]]],
        }
    ]

    data = json.dumps({"signature_name": "serving_default", "instances": inputs})
    url = "http://localhost:8501/v1/models/test_keras_model:predict"

    rest_outputs = predict_rest(data, url)

    print("REST outputs: ", rest_outputs)
    print("REST outputs shape: ", np.array(rest_outputs).shape)
