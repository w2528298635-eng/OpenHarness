from __future__ import annotations

from openharness.repopilot.swebench.gold import (
    extract_gold_files,
    extract_gold_labels,
)


def test_extract_gold_files_normalizes_rename_create_delete_and_test_paths() -> None:
    patch = """\
diff --git a/src/old.py b/src/new.py
similarity index 90%
rename from src/old.py
rename to src/new.py
diff --git a/dev/null b/src/created.py
new file mode 100644
--- /dev/null
+++ b/src/created.py
diff --git a/src/deleted.py b/src/deleted.py
deleted file mode 100644
--- a/src/deleted.py
+++ /dev/null
diff --git a/tests/test_hidden.py b/tests/test_hidden.py
--- a/tests/test_hidden.py
+++ b/tests/test_hidden.py
"""

    assert extract_gold_files(patch) == (
        "src/new.py",
        "src/created.py",
        "src/deleted.py",
    )


def test_extract_gold_files_normalizes_windows_separators() -> None:
    patch = """\
diff --git "a/src\\\\service.py" "b/src\\\\service.py"
--- "a/src\\\\service.py"
+++ "b/src\\\\service.py"
"""

    assert extract_gold_files(patch) == ("src/service.py",)


def test_extract_gold_labels_maps_changed_lines_to_smallest_python_symbol() -> None:
    base = """\
class Converter:
    def convert(self, value):
        normalized = value.strip()
        return normalized

async def fetch(value):
    return value
"""
    patched = """\
class Converter:
    def convert(self, value):
        normalized = value.strip().casefold()
        return normalized

async def fetch(value):
    return str(value)
"""
    patch = """\
diff --git a/src/service.py b/src/service.py
--- a/src/service.py
+++ b/src/service.py
@@ -1,7 +1,7 @@
 class Converter:
     def convert(self, value):
-        normalized = value.strip()
+        normalized = value.strip().casefold()
         return normalized
 
 async def fetch(value):
-    return value
+    return str(value)
"""

    labels = extract_gold_labels(
        patch,
        base_sources={"src/service.py": base},
        patched_sources={"src/service.py": patched},
    )

    assert labels.files == ("src/service.py",)
    assert labels.symbols == {
        "src/service.py": ("Converter.convert", "fetch"),
    }
    assert labels.symbol_denominator == 2


def test_extract_gold_labels_excludes_module_level_and_invalid_python_from_symbols() -> None:
    patch = """\
diff --git a/src/config.py b/src/config.py
--- a/src/config.py
+++ b/src/config.py
@@ -1 +1 @@
-TIMEOUT = 10
+TIMEOUT = 20
diff --git a/src/broken.py b/src/broken.py
--- a/src/broken.py
+++ b/src/broken.py
@@ -1 +1 @@
-def broken(
+def still_broken(
"""

    labels = extract_gold_labels(
        patch,
        base_sources={
            "src/config.py": "TIMEOUT = 10\n",
            "src/broken.py": "def broken(\n",
        },
        patched_sources={
            "src/config.py": "TIMEOUT = 20\n",
            "src/broken.py": "def still_broken(\n",
        },
    )

    assert labels.files == ("src/config.py", "src/broken.py")
    assert labels.symbols == {}
    assert labels.symbol_denominator == 0

