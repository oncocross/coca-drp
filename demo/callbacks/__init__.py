# callbacks/__init__.py

"""
This file makes the 'callbacks' directory a Python package and
exposes its main public function(s) for easy importing.
"""

# Expose the central event registration function from the event_listener module.
# This allows other scripts to import it directly with `from callbacks import ...`.
from .event_listener import register_event_listeners