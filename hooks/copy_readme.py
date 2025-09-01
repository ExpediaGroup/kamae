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

import os
import shutil


def on_startup(*args, **kwargs):
    # Copy the readme to the docs folder
    shutil.copy("README.md", "docs/README.md")


def on_shutdown(*args, **kwargs):
    # Remove the copied readme
    os.remove("docs/README.md")
