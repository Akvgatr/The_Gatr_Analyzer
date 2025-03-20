from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Retrieve a dictionary value using a key."""
    if isinstance(dictionary, dict):
        return dictionary.get(key, "")
    return ""
