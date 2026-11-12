"""StrawberryPicker — strawberry maturity and quality assessment from images."""

from strawberry_picker.images import register_heif

__version__ = "0.1.0"

register_heif()  # iPhone photos are HEIC; every Image.open in the package can read them
