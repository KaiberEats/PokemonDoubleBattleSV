import unittest
from collections import Counter

from train_lose import WEAK_PAIRS, build_schedule


class FocusScheduleTests(unittest.TestCase):
    def test_balanced_coverage_and_priority(self):
        names = tuple(f"team{i}" for i in range(1, 6))
        schedule = build_schedule(names, 10000, .5, 42)
        counts = Counter(schedule)
        self.assertEqual(len(schedule), 10000)
        self.assertEqual(len(counts), 20)
        self.assertTrue(all(a != b for a, b in schedule))
        self.assertTrue(all(counts[pair] in (1916, 1917) for pair in WEAK_PAIRS))
        self.assertTrue(all(n == 250 for pair, n in counts.items() if pair not in WEAK_PAIRS))
        self.assertEqual(schedule, build_schedule(names, 10000, .5, 42))
        self.assertNotEqual(schedule, build_schedule(names, 10000, .5, 43))

    def test_small_counts_and_ratio_limits(self):
        names = tuple(f"team{i}" for i in range(1, 6))
        for count in (1, 7, 53):
            for ratio in (0, .5, 1):
                schedule = build_schedule(names, count, ratio, 42)
                self.assertEqual(len(schedule), count)
                if ratio == 1:
                    self.assertTrue(set(schedule) <= set(WEAK_PAIRS))
        for ratio in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                build_schedule(names, 50, ratio, 42)


if __name__ == "__main__":
    unittest.main()
