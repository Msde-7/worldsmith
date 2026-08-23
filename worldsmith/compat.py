"""Test-only switch.

With deepslate compatibility on, the engine reproduces deepslate's known
deviations from the JVM so the conformance test can prove those are the only
differences. Rendering always runs with it off.
"""

COMPAT = {"deepslate": False}
