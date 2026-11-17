from concurrent.futures import ThreadPoolExecutor
import unittest
from performance_timing import PerformanceTimings


class TimingTests(unittest.TestCase):
    def test_records_all_threads_and_reset_is_atomic(self):
        timings = PerformanceTimings()

        @timings.timed('work')
        def work(_):
            return sum(range(50))

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(work, range(100)))
        row = timings.snapshot(reset=True)['work']
        self.assertEqual(row[0], 100)
        self.assertGreaterEqual(row[1], 0)
        self.assertGreaterEqual(row[2], 0)
        self.assertEqual(timings.snapshot(), {})

    def test_failed_call_is_recorded_and_exception_is_preserved(self):
        timings = PerformanceTimings()
        with self.assertRaisesRegex(ValueError, 'original'):
            with timings.measure('failed'):
                raise ValueError('original')
        self.assertEqual(timings.snapshot()['failed'][0], 1)

    def test_disabled_timer_does_not_record(self):
        timings = PerformanceTimings(enabled=False)
        with timings.measure('disabled'):
            pass
        self.assertEqual(timings.snapshot(), {})


if __name__ == '__main__':
    unittest.main()
