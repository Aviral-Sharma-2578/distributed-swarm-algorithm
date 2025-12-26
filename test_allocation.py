import unittest
from unittest.mock import MagicMock
import struct
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import SwarmAgent, AgentState, MsgType

class TestTaskAllocation(unittest.TestCase):
    def setUp(self):
        self.agent = SwarmAgent(agent_id=1, total_agents=3, capabilities=['extinguisher'])
        self.agent._send_msg = MagicMock()
        
    def test_calculate_utility_with_capability(self):
        # Task requires extinguisher, agent has it
        task = {'status': 'OPEN', 'pos': (1.0, 0.0), 'required_cap': 'extinguisher'}
        self.agent.position = [0.0, 0.0]
        
        util = self.agent._calculate_utility(task)
        # Dist = 1.0. U = 100 / (1+1) * 1.0 = 50.0
        self.assertAlmostEqual(util, 50.0)

    def test_calculate_utility_missing_capability(self):
        # Task requires sonar, agent has extinguisher
        task = {'status': 'OPEN', 'pos': (1.0, 0.0), 'required_cap': 'sonar'}
        self.agent.position = [0.0, 0.0]
        
        util = self.agent._calculate_utility(task)
        # Should be 0.0
        self.assertEqual(util, 0.0)

    def test_greedy_claim(self):
        # Utility 50.0 > 20.0 threshold
        self.agent.tasks = {101: {'status': 'OPEN', 'pos': (1.0, 0.0), 'required_cap': 'extinguisher'}}
        self.agent.position = [0.0, 0.0]
        
        self.agent._process_tasks()
        
        # Should initiate claim
        self.assertTrue(self.agent._send_msg.called)
        call_args = self.agent._send_msg.call_args[0]
        self.assertEqual(call_args[0], MsgType.TASK_CLAIM)
        
        # Verify payload contains TaskID=101 and Util=50.0
        payload = call_args[1]
        tid, util = struct.unpack('!If', payload)
        self.assertEqual(tid, 101)
        self.assertAlmostEqual(util, 50.0)

    def test_leader_conflict_resolution_win(self):
        # Agent is Leader
        self.agent.state = AgentState.LEADER
        self.agent.task_claims = {} # Empty start
        
        # Incoming claim for Task 101 with Utility 50.0 from Agent 2
        payload = struct.pack('!If', 101, 50.0)
        self.agent._handle_task_claim(sender=2, payload=payload)
        
        # Should award to Agent 2
        self.assertEqual(self.agent.task_claims[101]['winner'], 2)
        # Should broadcast CONFLICT msg saying 2 won
        self.agent._send_msg.assert_called()
        msg_type, out_payload = self.agent._send_msg.call_args[0]
        self.assertEqual(msg_type, MsgType.TASK_CONFLICT)
        tid, winner = struct.unpack('!IB', out_payload)
        self.assertEqual(winner, 2)

    def test_leader_hysteresis(self):
        # Agent is Leader
        self.agent.state = AgentState.LEADER
        # Task 101 already won by Agent 2 with Utility 50.0
        self.agent.task_claims = {101: {'winner': 2, 'utility': 50.0}}
        
        # Agent 3 claims with Utility 52.0 (Only +2.0, < 5.0 hysteresis)
        payload = struct.pack('!If', 101, 52.0)
        self.agent._handle_task_claim(sender=3, payload=payload)
        
        # Winner should remain 2
        self.assertEqual(self.agent.task_claims[101]['winner'], 2)
        # Should broadcast CONFLICT msg reaffirming Agent 2
        msg_type, out_payload = self.agent._send_msg.call_args[0]
        tid, winner = struct.unpack('!IB', out_payload)
        self.assertEqual(winner, 2)
        
        # Agent 3 claims with Utility 60.0 (+10.0, > 5.0 hysteresis)
        self.agent._send_msg.reset_mock()
        payload = struct.pack('!If', 101, 60.0)
        self.agent._handle_task_claim(sender=3, payload=payload)
        
        # Winner should switch to 3
        self.assertEqual(self.agent.task_claims[101]['winner'], 3)
        msg_type, out_payload = self.agent._send_msg.call_args[0]
        tid, winner = struct.unpack('!IB', out_payload)
        self.assertEqual(winner, 3)

if __name__ == '__main__':
    unittest.main()
