import time
import math
import argparse
import logging
import struct
import random
import enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - A%(name)s - %(levelname)s - %(message)s')

class MsgType(enum.IntEnum):
    HEARTBEAT = 0x01
    ELECTION_ACCLAIM = 0x02
    COORDINATOR = 0x03
    TASK_CLAIM = 0x04
    TASK_CONFLICT = 0x05

class AgentState(enum.Enum):
    FOLLOWER = 1
    ELECTION_WAIT = 2
    LEADER = 3

class SwarmAgent:
    def __init__(self, agent_id, total_agents, capabilities=None):
        self.agent_id = agent_id
        self.total_agents = total_agents
        self.logger = logging.getLogger(str(self.agent_id))
        
        # State
        self.state = AgentState.FOLLOWER
        self.leader_id = None
        self.leader_pos = None # (x, y)
        self.last_heartbeat_time = time.time()
        self.tick = 0
        
        # Election Jitter
        self.election_wait_start = 0.0
        self.election_delay = 0.0
        
        # Tasks: {task_id: {'status': 'OPEN'|'ASSIGNED'|'LOCKED', 'pos': (x,y), 'cap': str}}
        self.tasks = {} 
        # Claims: {task_id: {'winner': agent_id, 'utility': float}}
        self.task_claims = {} 

        # Physics & Sensors
        self.position = [0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.max_speed = 5.0 # m/s
        self.sensors = {'obstacles': [], 'neighbors': []} 
        self.target = None # (x, y)
        self.capabilities = capabilities if capabilities else []

        self.logger.info(f"Agent initialized. State: {self.state.name} Caps: {self.capabilities}")

    def set_target(self, x, y):
        self.target = (x, y)

    def update_sensors(self, obstacles, neighbors):
        """
        obstacles: list of (x, y, radius)
        neighbors: list of (id, x, y)
        """
        self.sensors['obstacles'] = obstacles
        self.sensors['neighbors'] = neighbors

    def update_loop(self):
        frequency = 10.0 # Hz
        period = 1.0 / frequency
        
        while True:
            start_time = time.time()
            self.tick += 1
            
            self._process_logic()
            self._update_physics(period)
            
            elapsed = time.time() - start_time
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_logic(self):
        # 1. Leader Election Logic
        self._check_election_timeout()
        
        # 2. Leader Duties
        if self.state == AgentState.LEADER:
            self._send_heartbeat()
            
        # 3. Task Logic
        self._process_tasks()

    def _update_physics(self, dt):
        # Formation Logic: Update target based on leader
        if self.state == AgentState.FOLLOWER and self.leader_pos:
            # Simple V-Shape: Odd IDs left, Even IDs right
            # Offset = ID * 2.0 meters back/side
            rank = self.agent_id
            
            # S3: Line Formation (Example)
            # x_offset = -rank * 2.0 (Behind leader)
            # y_offset = 0
            
            # S4: V-Shape
            x_offset = -2.0 * rank # Behind
            y_offset = 2.0 * rank if rank % 2 == 0 else -2.0 * rank # Side
            
            target_x = self.leader_pos[0] + x_offset
            target_y = self.leader_pos[1] + y_offset
            self.target = (target_x, target_y)

        if not self.target:
            return

        # Artificial Potential Fields
        # 1. Attractive Force to Target
        k_att = 1.0
        f_att = [0.0, 0.0]
        dist_to_target = math.sqrt((self.target[0] - self.position[0])**2 + 
                                   (self.target[1] - self.position[1])**2)
        
        if dist_to_target > 0.5: # Tolerance
            f_att[0] = k_att * (self.target[0] - self.position[0])
            f_att[1] = k_att * (self.target[1] - self.position[1])

        # 2. Repulsive Force from Obstacles
        k_rep = 50.0
        rho_0 = 5.0 # Influence radius
        f_rep = [0.0, 0.0]
        
        for obs in self.sensors['obstacles']:
            ox, oy, r = obs
            dist = math.sqrt((self.position[0] - ox)**2 + (self.position[1] - oy)**2) - r
            if dist <= 0.001: dist = 0.001 # Clamp
            
            if dist < rho_0:
                # F_rep = k_rep * (1/d - 1/rho0) * (1/d^2) * unit_vec
                # Simplified: push away
                mag = k_rep * (1.0/dist - 1.0/rho_0) / (dist**2)
                
                dx = self.position[0] - ox
                dy = self.position[1] - oy
                norm = math.sqrt(dx**2 + dy**2)
                f_rep[0] += (dx/norm) * mag
                f_rep[1] += (dy/norm) * mag

        # 3. Repulsive Force from Neighbors (Collision Avoidance)
        k_sep = 20.0
        f_sep = [0.0, 0.0]
        for n_id, nx, ny in self.sensors['neighbors']:
            dist = math.sqrt((self.position[0] - nx)**2 + (self.position[1] - ny)**2)
            if dist < 2.0: # personal space
                 if dist <= 0.001: dist = 0.001
                 mag = k_sep / (dist**2)
                 dx = self.position[0] - nx
                 dy = self.position[1] - ny
                 norm = math.sqrt(dx**2 + dy**2)
                 f_sep[0] += (dx/norm) * mag
                 f_sep[1] += (dy/norm) * mag

        # Sum Forces
        f_total = [f_att[0] + f_rep[0] + f_sep[0], f_att[1] + f_rep[1] + f_sep[1]]
        
        # Limit Force -> Acceleration -> Velocity
        # Simplified: Force ~ Velocity command for this agent type (Holonomic-ish)
        
        # Clamp Velocity
        v_mag = math.sqrt(f_total[0]**2 + f_total[1]**2)
        if v_mag > self.max_speed:
            scale = self.max_speed / v_mag
            self.velocity = [f_total[0] * scale, f_total[1] * scale]
        else:
            self.velocity = f_total

        # Integrate Position
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        
        if self.tick % 10 == 0:
            self.logger.info(f"Pos: ({self.position[0]:.2f}, {self.position[1]:.2f}) V: ({self.velocity[0]:.2f}, {self.velocity[1]:.2f})")

    # --- Communication Layer (Binary) ---
    def _pack_header(self, msg_type):
        # !B B I  (Type, Sender, Tick)
        return struct.pack('!BBI', msg_type, self.agent_id, self.tick)

    def _send_msg(self, msg_type, payload=b''):
        header = self._pack_header(msg_type)
        packet = header + payload
        # SIMULATION STUB: In real sim, this goes to UDP/TCP socket
        # self.sim_transport.send(packet)
        # For now, we mock receiving it locally if needed or just log
        pass 
        # self.logger.debug(f"SENT TYPE:{MsgType(msg_type).name} LEN:{len(packet)}")

    def on_message_received(self, data):
        if len(data) < 6:
            return

        header = data[:6]
        msg_type, sender, tick = struct.unpack('!BBI', header)
        payload = data[6:]

        if msg_type == MsgType.HEARTBEAT:
            self._handle_heartbeat(sender, payload) # Updated signature
        elif msg_type == MsgType.ELECTION_ACCLAIM:
            self._handle_election_acclaim(sender)
        elif msg_type == MsgType.COORDINATOR:
            self._handle_coordinator(sender)
        elif msg_type == MsgType.TASK_CLAIM:
            self._handle_task_claim(sender, payload)
        elif msg_type == MsgType.TASK_CONFLICT:
            self._handle_task_conflict(sender, payload)

    # --- Leader Election (Quiet Bully) ---
    def _check_election_timeout(self):
        if self.state == AgentState.LEADER:
            return

        # Failure Detection
        timeout = 3.0
        time_since_hb = time.time() - self.last_heartbeat_time
        
        if self.state == AgentState.FOLLOWER and time_since_hb > timeout:
            self.logger.warning(f"Leader timeout ({time_since_hb:.1f}s). Entering ELECTION_WAIT.")
            self.state = AgentState.ELECTION_WAIT
            self.election_wait_start = time.time()
            self.election_delay = random.uniform(0.0, 0.2) # Jitter
            self.leader_id = None
            self.leader_pos = None

        # Jitter Wait
        if self.state == AgentState.ELECTION_WAIT:
            if time.time() - self.election_wait_start > self.election_delay:
                # No one else took over during our delay? Claim it.
                self.logger.info("Election wait ended. Acclaiming Leadership.")
                self.state = AgentState.LEADER
                self.leader_id = self.agent_id
                self._send_msg(MsgType.ELECTION_ACCLAIM, struct.pack('!B', self.agent_id))
                self._send_msg(MsgType.COORDINATOR) # Announce win immediately (Optimistic)

    def _handle_heartbeat(self, sender, payload):
        if self.state == AgentState.LEADER and sender < self.agent_id:
            # We are higher ID, suppress them
             self._send_heartbeat() # Send our own heartbeat
             return
             
        if self.state == AgentState.LEADER and sender > self.agent_id:
            self.logger.info(f"Yielding to higher leader {sender}")
            self.state = AgentState.FOLLOWER

        self.leader_id = sender
        self.last_heartbeat_time = time.time()
        
        if len(payload) == 8:
            lx, ly = struct.unpack('!ff', payload)
            self.leader_pos = (lx, ly)
            
        if self.state == AgentState.ELECTION_WAIT:
            self.state = AgentState.FOLLOWER

    def _handle_election_acclaim(self, sender):
        if sender > self.agent_id:
            self.logger.info(f"Saw acclaim from higher node {sender}. Backing down.")
            self.state = AgentState.FOLLOWER
            self.leader_id = sender
            self.last_heartbeat_time = time.time() # Treat as liveness proof
        elif sender < self.agent_id and self.state in [AgentState.LEADER, AgentState.ELECTION_WAIT]:
            # We are higher, bully them back
            if self.state == AgentState.ELECTION_WAIT:
                 # If we were waiting, stop waiting and fight
                 self.state = AgentState.LEADER
                 self.leader_id = self.agent_id
            self._send_heartbeat()

    def _handle_coordinator(self, sender):
        self.leader_id = sender
        self.state = AgentState.FOLLOWER
        self.last_heartbeat_time = time.time()
        self.logger.info(f"New Coordinator: {sender}")

    def _send_heartbeat(self):
        # 1Hz Heartbeat
        # Payload: Leader Position (!ff)
        payload = struct.pack('!ff', self.position[0], self.position[1])
        
        if self.tick % 10 == 0:
            self._send_msg(MsgType.HEARTBEAT, payload)

    # --- Task Allocation (Greedy + Conflict Res) ---
    def _process_tasks(self):
        # 1. Greedy Claim Logic
        for task_id, task in self.tasks.items():
            if task['status'] == 'OPEN':
                utility = self._calculate_utility(task)
                if utility > 20.0:
                    self.logger.info(f"Claiming task {task_id} with U={utility:.1f}")
                    # Optimistic local claim
                    task['status'] = 'TENTATIVE' 
                    # Broadcast !If (TaskID, Utility)
                    self._send_msg(MsgType.TASK_CLAIM, struct.pack('!If', task_id, utility))

    def _handle_task_claim(self, sender, payload):
        task_id, utility = struct.unpack('!If', payload)
        
        # Leader Conflict Resolution
        if self.state == AgentState.LEADER:
            current_claim = self.task_claims.get(task_id)
            
            is_new_better = False
            if not current_claim:
                is_new_better = True
            else:
                # Hysteresis: New must be > Current + 5.0
                if utility > current_claim['utility'] + 5.0:
                    is_new_better = True
            
            if is_new_better:
                self.task_claims[task_id] = {'winner': sender, 'utility': utility}
                # Broadcast Conflict Resolution: TaskID, Winner
                self._send_msg(MsgType.TASK_CONFLICT, struct.pack('!IB', task_id, sender))
            elif current_claim and current_claim['winner'] != sender:
                 # Re-affirm current winner to stop the challenger
                 self._send_msg(MsgType.TASK_CONFLICT, struct.pack('!IB', task_id, current_claim['winner']))

    def _handle_task_conflict(self, sender, payload):
        task_id, winner_id = struct.unpack('!IB', payload)
        
        if winner_id == self.agent_id:
            self.logger.info(f"Won task {task_id}!")
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'ASSIGNED'
        else:
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'LOCKED'

    def _calculate_utility(self, task):
        # U = 100 / (1 + dist) * CapMatch
        dist = math.sqrt((self.position[0] - task['pos'][0])**2 + (self.position[1] - task['pos'][1])**2)
        
        # Capability Stub
        has_cap = 1.0
        if 'required_cap' in task and task['required_cap'] not in self.capabilities:
            has_cap = 0.0
            
        return (100.0 / (1.0 + dist)) * has_cap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Agent ID")
    parser.add_argument("--count", type=int, default=1, help="Total Agents")
    parser.add_argument("--caps", type=str, nargs='+', default=[], help="Agent Capabilities")
    args = parser.parse_args()
    
    agent = SwarmAgent(args.id, args.count, capabilities=args.caps)
    try:
        agent.update_loop()
    except KeyboardInterrupt:
        print("Shutting down.")
