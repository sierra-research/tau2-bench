import re
from typing import Dict


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """the target str contains tags in the format of <tag_name> ... </tag_name>, parse them out and return a dict

    Args:
        str_with_tags: The string to parse.

    Returns:
        A dictionary of tags and their contents.
    """

    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


if __name__ == "__main__":
    test_str = "<tag1>Hello</tag1> some text <tag2>World</tag2>"
    print(parse_tags(test_str))
