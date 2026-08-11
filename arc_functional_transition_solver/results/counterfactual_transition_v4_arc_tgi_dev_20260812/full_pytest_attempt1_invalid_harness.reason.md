# Full-suite attempt 1: invalid harness configuration

The first full-suite command set `PYTHONPATH=src`.  Three M04a test modules
import repository-local test fixtures as top-level modules, so collection
stopped with three `ModuleNotFoundError` exceptions before any test executed.

The retained stdout/stderr logs support no code-quality conclusion.  The retry
used the repository-required `PYTHONPATH=src:tests` and completed with 584
passed, three skipped, and six known M04a evidence-payload failures.
