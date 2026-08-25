import unittest
from wholebody.core.registry import Registry, MODELS


class TestRegistry(unittest.TestCase):

    def test_register_and_build(self):
        reg = Registry("test_registry")

        @reg.register("DummyModule")
        class Dummy:
            def __init__(self, val: int = 10):
                self.val = val

        obj = reg.build({"type": "DummyModule", "val": 42})
        self.assertIsInstance(obj, Dummy)
        self.assertEqual(obj.val, 42)

    def test_duplicate_registration_error(self):
        reg = Registry("test_dup")

        @reg.register("SameName")
        class A: pass

        with self.assertRaises(KeyError):
            @reg.register("SameName")
            class B: pass

    def test_typo_suggestion(self):
        reg = Registry("test_typo")

        @reg.register("HeatmapHead")
        class H: pass

        with self.assertRaises(KeyError) as ctx:
            reg.get("HeatmapHed")
        self.assertIn("Did you mean: HeatmapHead", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
