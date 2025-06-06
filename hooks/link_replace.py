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

import re


def on_page_markdown(markdown, page, config, files):
    # Regular expression to find markdown links
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def remove_py_ext(match):
        text = match.group(1)
        link = match.group(2)
        new_link = link.replace(".py", "")
        return f"[{text}]({new_link})"

    def remove_docs_in_path(match):
        text = match.group(1)
        link = match.group(2)
        new_link = link.replace("docs/", "")

        return f"[{text}]({new_link})"

    def add_reference_into_py_links(match):
        text = match.group(1)
        link = match.group(2)
        if ".py" in link:
            split_link = link.split("/")
            if split_link[0] == "src":
                split_link.insert(0, "reference")
            else:
                split_link.insert(1, "reference")
            new_link = "/".join(split_link)
        else:
            new_link = link
        return f"[{text}]({new_link})"

    new_markdown = pattern.sub(add_reference_into_py_links, markdown)
    new_markdown = pattern.sub(remove_py_ext, new_markdown)
    new_markdown = pattern.sub(remove_docs_in_path, new_markdown)

    return new_markdown
