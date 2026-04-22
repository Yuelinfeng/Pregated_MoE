import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts import analyze_prefetch_mechanisms as apm


TRACE_HEADER = (
    "event_type\ttrace_id\tcondition\trequest_id\tdomain\tstep_id\tsource_layer\ttarget_layer\t"
    "num_experts\tpredicted_experts\tactual_experts\tactual_counts\ttp\tfp\tfn\ttn\t"
    "prefetch_issue_id\texpert_id\tcache_hit\tready_before_consume\tstall_time_ms\n"
)


class PrefetchMechanismAnalysisTest(unittest.TestCase):
    def test_aggregate_prefetch_utility_splits_timely_and_useless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_root = Path(tmp_dir)
            trace_dir = results_root / "prefetch" / "stable_mixed" / "cr0.1"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "prefetch_trace.tsv"

            with trace_path.open("w", encoding="utf-8") as handle:
                handle.write(TRACE_HEADER)
                handle.write(
                    "CONFUSION\ttrace-0\tstable_mixed\treq-0\tmixed\t0\tdecoder::layer1\tdecoder::layer2\t"
                    "8\t1,2\t1,3\t1:4,3:2\t1\t1\t1\t5\t7\t-1\t-1\t-1\t-1\n"
                )
                handle.write(
                    "PREFETCH_EXPERT\ttrace-0\tstable_mixed\treq-0\tmixed\t0\tdecoder::layer1\tdecoder::layer2\t"
                    "-1\t\t\t\t-1\t-1\t-1\t-1\t7\t1\t1\t1\t0.3\n"
                )
                handle.write(
                    "PREFETCH_EXPERT\ttrace-0\tstable_mixed\treq-0\tmixed\t0\tdecoder::layer1\tdecoder::layer2\t"
                    "-1\t\t\t\t-1\t-1\t-1\t-1\t7\t2\t0\t0\t0.3\n"
                )

            summary, by_layer = apm.aggregate_prefetch_utility(results_root)

            self.assertEqual(len(summary), 1)
            row = summary.iloc[0]
            self.assertEqual(int(row["prefetch_experts"]), 2)
            self.assertEqual(int(row["timely_useful_prefetch_experts"]), 1)
            self.assertEqual(int(row["late_useless_prefetch_experts"]), 1)
            self.assertEqual(int(row["late_prefetch_experts"]), 1)
            self.assertAlmostEqual(float(row["avg_prefetch_stall_ms"]), 0.3, places=6)

            self.assertEqual(len(by_layer), 1)
            layer_row = by_layer.iloc[0]
            self.assertEqual(layer_row["source_layer"], "decoder::layer1")
            self.assertEqual(layer_row["target_layer"], "decoder::layer2")


if __name__ == "__main__":
    unittest.main()
