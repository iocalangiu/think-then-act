"""
Unit tests for think_then_act.policy.vlm_policy's pure text/action logic:
encode_action/decode_action and VLMPolicy's response parsers. None of this
touches torch/transformers (those are deferred imports inside VLMPolicy.__init__),
so it runs locally without a GPU.
"""

import numpy as np
import pytest

from think_then_act.policy.vlm_policy import N_BINS, VLMPolicy, decode_action, encode_action


@pytest.mark.parametrize(
    "bin_idx, expected",
    [(0, -1.0), (8, 0.0), (16, 1.0)],
)
def test_decode_action_known_bins(bin_idx, expected):
    assert decode_action(bin_idx) == pytest.approx(expected)


def test_encode_decode_round_trip_for_every_bin():
    for bin_idx in range(N_BINS):
        assert encode_action(decode_action(bin_idx)) == bin_idx


@pytest.mark.parametrize("value, expected_bin", [(-5.0, 0), (5.0, N_BINS - 1)])
def test_encode_action_clips_out_of_range_values(value, expected_bin):
    assert encode_action(value) == expected_bin


def test_extract_think_full_tags_present():
    text = "<think>gripper is above the block</think><action>8 8 8 8</action>"
    think_text, found = VLMPolicy._extract_think(text)
    assert found is True
    assert think_text == "gripper is above the block"


def test_extract_think_qwen2_style_missing_opening_tag():
    # Qwen2's chat template consumes the opening <think> into the prompt tokens,
    # so only content + </think> ever appears in the generated text.
    text = "gripper is above the block</think><action>8 8 8 8</action>"
    think_text, found = VLMPolicy._extract_think(text)
    assert found is True
    assert think_text == "gripper is above the block"


def test_extract_think_missing_tag_returns_empty():
    think_text, found = VLMPolicy._extract_think("no think tag here at all")
    assert found is False
    assert think_text == ""


def test_parse_action_valid_response():
    action, found = VLMPolicy._parse_action("<action>8 8 8 8</action>")
    assert found is True
    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, 0.0])


def test_parse_action_missing_tag_falls_back_to_zeros():
    action, found = VLMPolicy._parse_action("the model rambled without a tag")
    assert found is False
    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))


def test_parse_action_rejects_out_of_range_bin():
    # N_BINS == 17, so bin index 17 is one past the valid range [0, 16].
    action, found = VLMPolicy._parse_action("<action>17 0 0 0</action>")
    assert found is False
    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))


def test_parse_action_rejects_non_integer_bins():
    action, found = VLMPolicy._parse_action("<action>a b c d</action>")
    assert found is False
    np.testing.assert_array_equal(action, np.zeros(4, dtype=np.float32))
