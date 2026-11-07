"""
Integration test: Frame Processor + Graph Memory
Tests that the two modules work together on dummy data.
"""

import numpy as np
import unittest
from frame_processor import FrameProcessor
from graph_memory import GraphMemory


def test_frame_processor():
    """Test the frame processor on dummy frames."""
    print("=" * 60)
    print("TEST 1: Frame Processor")
    print("=" * 60)

    processor = FrameProcessor("test_game")

    # Create a dummy frame: 1x64x64 with values 0-12
    frame1 = np.random.randint(0, 13, size=(1, 64, 64))

    state1 = processor.process(frame1)
    print(f"\nFrame 1 processed:")
    print(f"  Hash: {state1['hash']}")
    print(f"  Number of blobs: {len(state1['blobs'])}")
    print(f"  First 3 blobs:")
    for i, blob in enumerate(state1['blobs'][:3]):
        print(f"    Blob {i}: size={blob['size']}, value={blob['primary_value']}, tier={blob['priority_tier']}")

    # Process another frame
    frame2 = np.random.randint(0, 13, size=(1, 64, 64))
    state2 = processor.process(frame2)
    print(f"\nFrame 2 processed:")
    print(f"  Hash: {state2['hash']}")
    print(f"  Number of blobs: {len(state2['blobs'])}")

    # Test that different frames have different hashes
    assert state1['hash'] != state2['hash'], "Different frames should have different hashes"
    print("\n[OK] Frame processor working: different frames -> different hashes")


def test_graph_memory():
    """Test the graph memory on dummy transitions."""
    print("\n" + "=" * 60)
    print("TEST 2: Graph Memory")
    print("=" * 60)

    graph = GraphMemory()

    # Simulate some state transitions
    print("\nRecording transitions...")
    graph.record_transition("state_a", "action_0", "state_b")
    graph.record_transition("state_b", "action_1", "state_c")
    graph.record_transition("state_c", "action_2", "state_a")  # creates a loop

    stats = graph.get_stats()
    print(f"\nAfter 3 transitions:")
    print(f"  States: {stats['num_states']}")
    print(f"  Actions recorded: {stats['total_actions']}")
    print(f"  Actions tested: {stats['tested_actions']}")
    print(f"  Steps taken: {stats['steps_taken']}")

    # Test frontier detection
    frontier = graph.get_frontier("state_a")
    print(f"\nFrontier from state_a: {frontier}")

    # Test tagging high-value edges
    graph.tag_high_value_edge("state_b", "action_1")
    print(f"Tagged (state_b, action_1) as high-value")

    # Test loop detection
    print(f"\nTesting loop detection after 3 steps...")
    is_loop = graph.is_loop_detected("state_a", window_size=3)
    print(f"  Loop detected? {is_loop}")

    print("\n[OK] Graph memory working: transitions recorded and frontier detected")


def test_shallowest_frontier():
    """A RESET replay target must be the shallowest explored frontier."""
    graph = GraphMemory()
    for state in ("root", "near", "deep", "far", "mid"):
        graph.add_state(state)
    graph.nodes["root"].add_action((1, None, None), 0)
    graph.nodes["root"].add_action((2, None, None), 0)
    graph.record_transition("root", (1, None, None), "near")
    graph.record_transition("near", (3, None, None), "deep")
    graph.record_transition("deep", (4, None, None), "far")
    graph.nodes["far"].add_action((5, None, None), 0)
    graph.nodes["mid"].add_action((6, None, None), 0)
    # root has one untested edge, which is shallower than far.
    target, path = graph.get_shallowest_frontier("root")
    assert target == "root"
    assert path == ["root"]
    # Once root is exhausted, BFS selects the depth-1 frontier before deeper.
    graph.record_transition("root", (2, None, None), "mid")
    target, path = graph.get_shallowest_frontier("root")
    assert target == "mid"
    assert path == ["root", "mid"]


def test_replay_does_not_count_as_a_new_state_action_trial():
    graph = GraphMemory()
    graph.add_state("root").add_action((1, None, None), 0)
    graph.record_transition("root", (1, None, None), "child")
    graph.record_transition("root", (1, None, None), "child", replayed=True)
    edge = graph.nodes["root"].actions[(1, None, None)]
    assert edge.tested
    assert edge.times_tested == 1
    assert edge.replay_count == 1
    assert graph.action_outcome_counts(
        (1, None, None), level_id=0
    )["attempts"] == 1


class FrontierBehaviorTests(unittest.TestCase):
    def test_shallowest_frontier_order(self):
        test_shallowest_frontier()

    def test_replay_is_not_a_new_trial(self):
        test_replay_does_not_count_as_a_new_state_action_trial()


def test_integration():
    """Test frame processor and graph memory together."""
    print("\n" + "=" * 60)
    print("TEST 3: Integration (Frame Processor + Graph Memory)")
    print("=" * 60)

    processor = FrameProcessor("ls20")
    graph = GraphMemory()

    print("\nSimulating a short episode:")

    # Simulate 5 steps
    for step in range(5):
        # Generate a random frame
        frame = np.random.randint(0, 13, size=(1, 64, 64))

        # Process it
        state = processor.process(frame)
        state_hash = state['hash']

        # Register blobs as actions
        graph.get_or_create_action_from_blobs(state_hash, state['blobs'])

        print(f"\nStep {step}:")
        print(f"  State hash: {state_hash[:8]}...")
        print(f"  Blobs found: {len(state['blobs'])}")
        print(f"  Actions registered: {len(graph.nodes[state_hash].actions)}")

        # Simulate taking an action to the next state
        if step < 4:
            next_frame = np.random.randint(0, 13, size=(1, 64, 64))
            next_state = processor.process(next_frame)
            next_hash = next_state['hash']

            # Pick a random action (blob)
            action_id = f"blob_0"
            graph.record_transition(state_hash, action_id, next_hash)

    # Print final stats
    print("\n" + "-" * 60)
    final_stats = graph.get_stats()
    print("Final graph statistics:")
    for key, val in final_stats.items():
        print(f"  {key}: {val}")

    print("\n[OK] Integration test passed: frames processed and graph maintained")


if __name__ == "__main__":
    test_frame_processor()
    test_graph_memory()
    test_integration()
    test_shallowest_frontier()
    test_replay_does_not_count_as_a_new_state_action_trial()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
