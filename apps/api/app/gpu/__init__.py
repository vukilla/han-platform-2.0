"""GPU-only helpers.

These modules are safe to import on the Mac control-plane containers because they avoid importing
GPU-heavy dependencies (torch / isaaclab) at module import time. GPU workers can execute them.
"""

