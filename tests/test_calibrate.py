"""calibrate — 온도 찾기 · 문턱 둘 고르기 (설계 7절)."""
import math
import unittest

import helpers  # noqa: F401

import calibrate as cal


def logits_for(probs):
    return [math.log(p) for p in probs]


class TemperatureTest(unittest.TestCase):
    def test_overconfident_logits_get_t_above_one(self):
        # 확신 0.99 인데 절반만 맞음 → T 를 키워 확신을 낮춰야 NLL 이 준다
        rows = []
        for i in range(40):
            label = 0 if i % 2 == 0 else 1
            rows.append({"label": label, "logits": logits_for([0.99, 0.005, 0.005])})
        t = cal.fit_temperature(rows)
        self.assertGreater(t, 1.0)
        self.assertLess(cal.nll(rows, t), cal.nll(rows, 1.0))

    def test_grid_bounds(self):
        self.assertEqual(cal.T_GRID[0], 0.5)
        self.assertEqual(cal.T_GRID[-1], 3.0)


class ThresholdTest(unittest.TestCase):
    def test_support_threshold_is_smallest_meeting_two_percent(self):
        scored = []
        scored += [([0.95, 0.03, 0.02], 0)] * 50          # 진짜 지지
        scored += [([0.03, 0.95, 0.02], 1)] * 50          # 진짜 반대 (확신)
        scored += [([0.80, 0.10, 0.10], 2)] * 2           # 거짓 지지 0.80
        scored += [([0.70, 0.20, 0.10], 1)] * 3           # 거짓 지지 0.70
        # 지지 아닌 쌍 55 → 허용 floor(1.1) = 1. 0.70 이하면 5 · 0.71~0.80 이면 2 · 0.81 부터 0
        got = cal.pick_thresholds(scored)
        self.assertEqual(got["dev_false_support_allowed"], 1)
        self.assertEqual(got["nli_sure_support"], 0.81)
        self.assertEqual(got["dev_false_support"], 0)

    def test_sure_needs_ninety_percent(self):
        scored = []
        scored += [([0.1, 0.85, 0.05], 1)] * 9            # 맞은 B 0.85
        scored += [([0.1, 0.60, 0.30], 2)] * 5            # 틀린 B 0.60
        scored += [([0.9, 0.05, 0.05], 0)] * 10
        got = cal.pick_thresholds(scored)
        # 0.60 까지는 정확도 9/14 → 0.61 부터 9/9
        self.assertEqual(got["nli_sure"], 0.61)

    def test_unsure_over_limit_lowers_only_sure(self):
        scored = [([0.03, 0.95, 0.02], 1)] * 20             # 확신 맞은 B
        scored += [([0.05, 0.55, 0.40], 1)] * 10            # 0.55 맞은 B
        scored += [([0.05, 0.55, 0.40], 2)] * 10            # 0.55 틀린 B
        scored += [([0.95, 0.03, 0.02], 0)] * 10
        # 정확도 자로는 0.56 (보류 20/50 = 40%) → 30% 안으로 0.55 까지 내린다. 지지 문턱은 그대로
        got = cal.pick_thresholds(scored)
        self.assertEqual(got["nli_sure_lowered_from"], 0.56)
        self.assertEqual(got["nli_sure"], 0.55)
        self.assertLessEqual(got["dev_unsure_rate"], 0.30)
        self.assertEqual(got["nli_sure_support"], 0.5)

    def test_go_tie_gap_counts_as_unsure(self):
        self.assertFalse(cal.is_sure([0.495, 0.495, 0.01], 0.4, 0.4))


if __name__ == "__main__":
    unittest.main()
