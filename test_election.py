import unittest
import time
from unittest.mock import MagicMock
import struct
import sys
import os

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import SwarmAgent, AgentState, MsgType

class TestSwarmAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SwarmAgent(agent_id=1, total_agents=3)
        # Mock send_msg to avoid actual socket/stubs issues
        self.agent._send_msg = MagicMock()
        
    def test_initial_state(self):
        self.assertEqual(self.agent.state, AgentState.FOLLOWER)
        self.assertIsNone(self.agent.leader_id)

    def test_election_timeout_trigger(self):
        """Test that missing heartbeats triggers election wait."""
        # Fast forward time
        self.agent.last_heartbeat_time = time.time() - 5.0 # > 3.0s timeout
        
        self.agent._check_election_timeout()
        
        self.assertEqual(self.agent.state, AgentState.ELECTION_WAIT)
        self.assertIsNotNone(self.agent.election_wait_start)

    def test_election_victory_after_wait(self):
        """Test that after waiting, agent claims leadership."""
        self.agent.state = AgentState.ELECTION_WAIT
        self.agent.election_wait_start = time.time() - 1.0 # > 0.2s jitter
        self.agent.election_delay = 0.1
        
        self.agent._check_election_timeout()
        
        self.assertEqual(self.agent.state, AgentState.LEADER)
        self.assertEqual(self.agent.leader_id, 1)
        # Should send ACCLAIM and COORDINATOR
        self.assertTrue(self.agent._send_msg.called)
        calls = [c[0][0] for c in self.agent._send_msg.call_args_list]
        self.assertIn(MsgType.ELECTION_ACCLAIM, calls)
        self.assertIn(MsgType.COORDINATOR, calls)

    def test_submission_to_higher_id(self):
        """Test that agent backs down when seeing higher ID acclaim."""
        self.agent.state = AgentState.LEADER
        self.agent.agent_id = 1
        
        # Higher ID (2) acclaims
        self.agent._handle_election_acclaim(sender=2)
        
        self.assertEqual(self.agent.state, AgentState.FOLLOWER)
        self.assertEqual(self.agent.leader_id, 2)

    def test_bullying_lower_id(self):
        """Test that agent fights back against lower ID."""
        self.agent.state = AgentState.LEADER
        self.agent.agent_id = 2
        
        # Lower ID (1) acclaims
        self.agent._handle_election_acclaim(sender=1)
        
        # Should stay Leader and send Heartbeat to suppress
        self.assertEqual(self.agent.state, AgentState.LEADER)
        # Heartbeat now includes 8 bytes of position data (0.0, 0.0) -> 8 null bytes
        expected_payload = struct.pack('!ff', 0.0, 0.0)
        self.agent._send_msg.assert_called_with(MsgType.HEARTBEAT, expected_payload)

if __name__ == '__main__':
    unittest.main()
