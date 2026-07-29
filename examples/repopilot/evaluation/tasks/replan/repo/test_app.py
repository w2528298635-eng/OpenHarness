from app import slugify


def test_spaces_and_underscores():
    assert slugify("Hello_big World") == "hello-big-world"
