import importlib.util, sys, traceback, pathlib

root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root))
passed = failed = 0
for f in sorted((root / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            if callable(fn):
                try:
                    fn(); passed += 1
                    print(f"PASS  {f.stem}::{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL  {f.stem}::{name}")
                    traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
