"""Live inspection demo: a local web page that grades a photo of strawberries.

``summary`` turns per-berry predictions into a batch verdict; ``server`` serves the page
in ``page/`` and a small JSON API around a background-loaded predictor.
"""

from strawberry_picker.demo.summary import summarize

__all__ = ["summarize"]
