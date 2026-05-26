import os

# Allow importing modules from the repository root using the orbit_wars_ai namespace.
# This makes `import orbit_wars_ai.environment.wrapper` work even though the code lives
# in top-level `environment/` and `agents/` directories.
__path__.append(os.path.dirname(os.path.dirname(__file__)))
