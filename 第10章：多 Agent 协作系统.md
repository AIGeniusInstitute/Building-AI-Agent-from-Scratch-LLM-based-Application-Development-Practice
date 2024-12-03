# 第四部分：AI Agent 高级主题

# 第10章：多 Agent 协作系统

多 Agent 协作系统是由多个智能 Agent 组成的网络，这些 Agent 共同工作以解决复杂问题或完成大规模任务。本章将探讨如何设计、实现和优化这种系统。

## 10.1 多 Agent 系统架构

### 10.1.1 集中式 vs 分布式架构

多 Agent 系统可以采用集中式或分布式架构，每种架构都有其优缺点。

示例（简化的多 Agent 系统架构比较）：

```python
from abc import ABC, abstractmethod

class AgentSystem(ABC):
    @abstractmethod
    def add_agent(self, agent):
        pass

    @abstractmethod
    def remove_agent(self, agent_id):
        pass

    @abstractmethod
    def communicate(self, sender_id, receiver_id, message):
        pass

    @abstractmethod
    def execute_task(self, task):
        pass

class CentralizedSystem(AgentSystem):
    def __init__(self):
        self.agents = {}
        self.central_controller = CentralController(self)

    def add_agent(self, agent):
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id):
        del self.agents[agent_id]

    def communicate(self, sender_id, receiver_id, message):
        self.central_controller.route_message(sender_id, receiver_id, message)

    def execute_task(self, task):
        return self.central_controller.assign_and_execute(task)

class DistributedSystem(AgentSystem):
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent
        for other_agent in self.agents.values():
            if other_agent.id != agent.id:
                agent.add_peer(other_agent)
                other_agent.add_peer(agent)

    def remove_agent(self, agent_id):
        removed_agent = self.agents.pop(agent_id)
        for agent in self.agents.values():
            agent.remove_peer(agent_id)

    def communicate(self, sender_id, receiver_id, message):
        if sender_id in self.agents and receiver_id in self.agents:
            self.agents[sender_id].send_message(receiver_id, message)

    def execute_task(self, task):
        # In a distributed system, any agent can initiate task execution
        initiator = next(iter(self.agents.values()))
        return initiator.initiate_task(task)

class CentralController:
    def __init__(self, system):
        self.system = system

    def route_message(self, sender_id, receiver_id, message):
        if receiver_id in self.system.agents:
            self.system.agents[receiver_id].receive_message(sender_id, message)

    def assign_and_execute(self, task):
        # Simple round-robin task assignment
        assigned_agent = next(iter(self.system.agents.values()))
        return assigned_agent.execute_task(task)

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.peers = {}

    def add_peer(self, peer):
        self.peers[peer.id] = peer

    def remove_peer(self, peer_id):
        del self.peers[peer_id]

    def send_message(self, receiver_id, message):
        if receiver_id in self.peers:
            self.peers[receiver_id].receive_message(self.id, message)

    def receive_message(self, sender_id, message):
        print(f"Agent {self.id} received message from Agent {sender_id}: {message}")

    def execute_task(self, task):
        print(f"Agent {self.id} executing task: {task}")
        # Implement task execution logic here
        return f"Task '{task}' completed by Agent {self.id}"

    def initiate_task(self, task):
        # In a distributed system, the initiating agent might need to coordinate with peers
        print(f"Agent {self.id} initiating task: {task}")
        return self.execute_task(task)

# 使用示例
centralized_system = CentralizedSystem()
distributed_system = DistributedSystem()

for i in range(3):
    centralized_system.add_agent(Agent(f"C{i}"))
    distributed_system.add_agent(Agent(f"D{i}"))

print("Centralized System:")
centralized_system.execute_task("Analyze data")
centralized_system.communicate("C0", "C1", "Hello from C0")

print("\nDistributed System:")
distributed_system.execute_task("Process images")
distributed_system.communicate("D0", "D1", "Hello from D0")
```

### 10.1.2 角色定义与分工

在多 Agent 系统中，明确定义每个 Agent 的角色和职责是至关重要的。

示例（基于角色的多 Agent 系统）：

```python
from enum import Enum

class AgentRole(Enum):
    COORDINATOR = 1
    EXECUTOR = 2
    MONITOR = 3

class RoleBasedAgent(Agent):
    def __init__(self, agent_id, role):
        super().__init__(agent_id)
        self.role = role

    def perform_role_action(self, task):
        if self.role == AgentRole.COORDINATOR:
            return self.coordinate_task(task)
        elif self.role == AgentRole.EXECUTOR:
            return self.execute_task(task)
        elif self.role == AgentRole.MONITOR:
            return self.monitor_task(task)

    def coordinate_task(self, task):
        print(f"Agent {self.id} coordinating task: {task}")
        # Implement coordination logic
        return f"Task '{task}' coordinated"

    def execute_task(self, task):
        print(f"Agent {self.id} executing task: {task}")
        # Implement execution logic
        return f"Task '{task}' executed"

    def monitor_task(self, task):
        print(f"Agent {self.id} monitoring task: {task}")
        # Implement monitoring logic
        return f"Task '{task}' monitored"

class RoleBasedSystem(AgentSystem):
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent):
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id):
        del self.agents[agent_id]

    def communicate(self, sender_id, receiver_id, message):
        if sender_id in self.agents and receiver_id in self.agents:
            self.agents[sender_id].send_message(receiver_id, message)

    def execute_task(self, task):
        coordinator = next(agent for agent in self.agents.values() if agent.role == AgentRole.COORDINATOR)
        executor = next(agent for agent in self.agents.values() if agent.role == AgentRole.EXECUTOR)
        monitor = next(agent for agent in self.agents.values() if agent.role == AgentRole.MONITOR)

        coordination_result = coordinator.perform_role_action(task)
        execution_result = executor.perform_role_action(task)
        monitoring_result = monitor.perform_role_action(task)

        return f"{coordination_result}, {execution_result}, {monitoring_result}"

# 使用示例
role_based_system = RoleBasedSystem()
role_based_system.add_agent(RoleBasedAgent("A1", AgentRole.COORDINATOR))
role_based_system.add_agent(RoleBasedAgent("A2", AgentRole.EXECUTOR))
role_based_system.add_agent(RoleBasedAgent("A3", AgentRole.MONITOR))

result = role_based_system.execute_task("Complex data analysis")
print(f"Task execution result: {result}")
```

### 10.1.3 通信协议设计

设计一个高效、可靠的通信协议对于多 Agent 系统的性能至关重要。

示例（简单的 Agent 间通信协议）：

```python
import json
from enum import Enum

class MessageType(Enum):
    TASK_ASSIGNMENT = 1
    TASK_RESULT = 2
    STATUS_UPDATE = 3
    QUERY = 4
    RESPONSE = 5

class Message:
    def __init__(self, sender, receiver, msg_type, content):
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type
        self.content = content

    def to_json(self):
        return json.dumps({
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.name,
            "content": self.content
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(
            data["sender"],
            data["receiver"],
            MessageType[data["type"]],
            data["content"]
        )

class CommunicationProtocol:
    @staticmethod
    def send_message(sender, receiver, msg_type, content):
        message = Message(sender, receiver, msg_type, content)
        # In a real system, this would use actual network communication
        print(f"Sending: {message.to_json()}")
        return message.to_json()

    @staticmethod
    def receive_message(json_str):
        message = Message.from_json(json_str)
        print(f"Received: {json_str}")
        return message

class CommunicatingAgent(Agent):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.protocol = CommunicationProtocol()

    def send_task_assignment(self, receiver, task):
        return self.protocol.send_message(self.id, receiver, MessageType.TASK_ASSIGNMENT, task)

    def send_task_result(self, receiver, result):
        return self.protocol.send_message(self.id, receiver, MessageType.TASK_RESULT, result)

    def send_status_update(self, receiver, status):
        return self.protocol.send_message(self.id, receiver, MessageType.STATUS_UPDATE, status)

    def send_query(self, receiver, query):
        return self.protocol.send_message(self.id, receiver, MessageType.QUERY, query)

    def send_response(self, receiver, response):
        return self.protocol.send_message(self.id, receiver, MessageType.RESPONSE, response)

    def receive_message(self, json_str):
        message = self.protocol.receive_message(json_str)
        if message.type == MessageType.TASK_ASSIGNMENT:
            self.handle_task_assignment(message.content)
        elif message.type == MessageType.QUERY:
            self.handle_query(message.sender, message.content)
        # Handle other message types...

    def handle_task_assignment(self, task):
        print(f"Agent {self.id} received task assignment: {task}")
        # Implement task handling logic

    def handle_query(self, sender, query):
        print(f"Agent {self.id} received query from {sender}: {query}")
        # Implement query handling logic

# 使用示例
agent1 = CommunicatingAgent("A1")
agent2 = CommunicatingAgent("A2")

# Agent1 assigns a task to Agent2
task_msg = agent1.send_task_assignment("A2", "Analyze dataset X")

# Agent2 receives and processes the message
agent2.receive_message(task_msg)

# Agent2 sends a query to Agent1
query_msg = agent2.send_query("A1", "Need more info about dataset X")

# Agent1 receives and processes the query
agent1.receive_message(query_msg)
```

这些示例展示了多 Agent 系统的基本架构、角色定义和通信协议设计。在实际应用中，这些系统会更加复杂，可能包括更多的角色类型、更复杂的任务分配算法、更健壮的错误处理机制，以及更高效的通信协议。此外，还需要考虑安全性、可扩展性、容错能力等方面，以构建一个真正强大和可靠的多 Agent 协作系统。

## 10.2 任务分配与协调

在多 Agent 系统中，有效的任务分配和协调机制对于系统的整体性能至关重要。

### 10.2.1 任务分解策略

复杂任务需要被分解成更小、更易管理的子任务，以便于分配给不同的 Agent。

示例（任务分解器）：

```python
from typing import List, Dict

class Task:
    def __init__(self, task_id: str, description: str, complexity: int):
        self.task_id = task_id
        self.description = description
        self.complexity = complexity
        self.subtasks: List[Task] = []

    def add_subtask(self, subtask: 'Task'):
        self.subtasks.append(subtask)

class TaskDecomposer:
    def __init__(self, complexity_threshold: int):
        self.complexity_threshold = complexity_threshold

    def decompose(self, task: Task) -> List[Task]:
        if task.complexity <= self.complexity_threshold:
            return [task]
        
        subtasks = []
        # 这里应该实现实际的任务分解逻辑
        # 简单起见，我们只是创建两个相等的子任务
        subtask1 = Task(f"{task.task_id}_1", f"Subtask 1 of {task.description}", task.complexity // 2)
        subtask2 = Task(f"{task.task_id}_2", f"Subtask 2 of {task.description}", task.complexity // 2)
        
        task.add_subtask(subtask1)
        task.add_subtask(subtask2)
        
        return self.decompose(subtask1) + self.decompose(subtask2)

# 使用示例
decomposer = TaskDecomposer(complexity_threshold=5)
main_task = Task("MAIN", "Analyze large dataset", 10)
subtasks = decomposer.decompose(main_task)

print(f"Main task '{main_task.description}' decomposed into {len(subtasks)} subtasks:")
for subtask in subtasks:
    print(f"- {subtask.task_id}: {subtask.description} (Complexity: {subtask.complexity})")
```

### 10.2.2 负载均衡算法

负载均衡确保任务被公平地分配给所有可用的 Agent，避免某些 Agent 过载而其他 Agent 闲置。

示例（简单的负载均衡器）：

```python
import heapq

class Agent:
    def __init__(self, agent_id: str, capacity: int):
        self.agent_id = agent_id
        self.capacity = capacity
        self.current_load = 0

    def can_handle(self, task: Task) -> bool:
        return self.current_load + task.complexity <= self.capacity

    def assign_task(self, task: Task):
        if self.can_handle(task):
            self.current_load += task.complexity
            return True
        return False

class LoadBalancer:
    def __init__(self):
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def get_available_agent(self, task: Task) -> Agent:
        available_agents = [agent for agent in self.agents if agent.can_handle(task)]
        if not available_agents:
            return None
        return min(available_agents, key=lambda a: a.current_load)

    def assign_tasks(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        assignment = {agent.agent_id: [] for agent in self.agents}
        task_heap = [(task.complexity, task) for task in tasks]
        heapq.heapify(task_heap)

        while task_heap:
            _, task = heapq.heappop(task_heap)
            agent = self.get_available_agent(task)
            if agent:
                agent.assign_task(task)
                assignment[agent.agent_id].append(task)
            else:
                print(f"Warning: Unable to assign task {task.task_id}")

        return assignment

# 使用示例
balancer = LoadBalancer()
balancer.add_agent(Agent("A1", capacity=10))
balancer.add_agent(Agent("A2", capacity=15))
balancer.add_agent(Agent("A3", capacity=20))

tasks = [
    Task("T1", "Task 1", 5),
    Task("T2", "Task 2", 8),
    Task("T3", "Task 3", 3),
    Task("T4", "Task 4", 7),
    Task("T5", "Task 5", 4)
]

assignment = balancer.assign_tasks(tasks)

for agent_id, assigned_tasks in assignment.items():
    print(f"Agent {agent_id} assigned tasks:")
    for task in assigned_tasks:
        print(f"- {task.task_id}: {task.description} (Complexity: {task.complexity})")
```

### 10.2.3 冲突检测与解决

在多 Agent 系统中，资源竞争和任务冲突是常见问题。需要实现机制来检测和解决这些冲突。

示例（简单的冲突检测与解决系统）：

```python
from typing import Set

class Resource:
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self.locked_by: Set[str] = set()

    def is_available(self) -> bool:
        return len(self.locked_by) == 0

    def lock(self, agent_id: str) -> bool:
        if self.is_available():
            self.locked_by.add(agent_id)
            return True
        return False

    def unlock(self, agent_id: str):
        self.locked_by.discard(agent_id)

class ConflictResolver:
    def __init__(self):
        self.resources: Dict[str, Resource] = {}

    def add_resource(self, resource: Resource):
        self.resources[resource.resource_id] = resource

    def request_resources(self, agent_id: str, resource_ids: List[str]) -> bool:
        available_resources = []
        for resource_id in resource_ids:
            resource = self.resources.get(resource_id)
            if resource and resource.is_available():
                available_resources.append(resource)
            else:
                # 如果有任何资源不可用，释放已锁定的资源
                for res in available_resources:
                    res.unlock(agent_id)
                return False
        
        # 所有请求的资源都可用，锁定它们
        for resource in available_resources:
            resource.lock(agent_id)
        return True

    def release_resources(self, agent_id: str, resource_ids: List[str]):
        for resource_id in resource_ids:
            resource = self.resources.get(resource_id)
            if resource:
                resource.unlock(agent_id)

# 使用示例
resolver = ConflictResolver()
resolver.add_resource(Resource("R1"))
resolver.add_resource(Resource("R2"))
resolver.add_resource(Resource("R3"))

print("Agent A requesting resources R1 and R2:")
if resolver.request_resources("A", ["R1", "R2"]):
    print("Resources allocated to Agent A")
else:
    print("Failed to allocate resources to Agent A")

print("\nAgent B requesting resources R2 and R3:")
if resolver.request_resources("B", ["R2", "R3"]):
    print("Resources allocated to Agent B")
else:
    print("Failed to allocate resources to Agent B")

print("\nAgent A releasing resources:")
resolver.release_resources("A", ["R1", "R2"])

print("\nAgent B requesting resources R2 and R3 again:")
if resolver.request_resources("B", ["R2", "R3"]):
    print("Resources allocated to Agent B")
else:
    print("Failed to allocate resources to Agent B")
```

## 10.3 知识共享与同步

在多 Agent 系统中，有效的知识共享和同步机制对于保持系统的一致性和提高整体性能至关重要。

### 10.3.1 分布式知识库

设计一个分布式知识库，允许多个 Agent 共享和访问信息。

示例（简单的分布式知识库）：

```python
from typing import Any, Dict
import threading

class DistributedKnowledgeBase:
    def __init__(self):
        self.knowledge: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def add_knowledge(self, key: str, value: Any):
        with self.lock:
            self.knowledge[key] = value

    def get_knowledge(self, key: str) -> Any:
        with self.lock:
            return self.knowledge.get(key)

    def update_knowledge(self, key: str, value: Any) -> bool:
        with self.lock:
            if key in self.knowledge:
                self.knowledge[key] = value
                return True
            return False

    def remove_knowledge(self, key: str) -> bool:
        with self.lock:
            if key in self.knowledge:
                del self.knowledge[key]
                return True
            return False

class KnowledgeSharingAgent(Agent):
    def __init__(self, agent_id: str, knowledge_base: DistributedKnowledgeBase):
        super().__init__(agent_id)
        self.knowledge_base = knowledge_base

    def share_knowledge(self, key: str, value: Any):
        self.knowledge_base.add_knowledge(key, value)
        print(f"Agent {self.agent_id} shared knowledge: {key} = {value}")

    def retrieve_knowledge(self, key: str) -> Any:
        value = self.knowledge_base.get_knowledge(key)
        print(f"Agent {self.agent_id} retrieved knowledge: {key} = {value}")
        return value

    def update_shared_knowledge(self, key: str, value: Any) -> bool:
        success = self.knowledge_base.update_knowledge(key, value)
        if success:
            print(f"Agent {self.agent_id} updated shared knowledge: {key} = {value}")
        else:
            print(f"Agent {self.agent_id} failed to update knowledge: {key}")
        return success

# 使用示例
shared_kb = DistributedKnowledgeBase()

agent1 = KnowledgeSharingAgent("A1", shared_kb)
agent2 = KnowledgeSharingAgent("A2", shared_kb)

agent1.share_knowledge("weather", "sunny")
agent2.retrieve_knowledge("weather")
agent2.update_shared_knowledge("weather", "rainy")
agent1.retrieve_knowledge("weather")
```

### 10.3.2 知识一致性维护

实现机制以确保多个 Agent 之间的知识保持一致。

示例（使用版本控制的知识一致性维护）：

```python
from dataclasses import dataclass
from typing import Any, Dict, Tuple

@dataclass
class VersionedKnowledge:
    value: Any
    version: int

class ConsistentKnowledgeBase:
    def __init__(self):
        self.knowledge: Dict[str, VersionedKnowledge] = {}
        self.lock = threading.Lock()

    def add_or_update_knowledge(self, key: str, value: Any) -> int:
        with self.lock:
            if key in self.knowledge:
                current_version = self.knowledge[key].version
                new_version = current_version + 1
            else:
                new_version = 1
            self.knowledge[key] = VersionedKnowledge(value, new_version)
            return new_version

    def get_knowledge(self, key: str) -> Tuple[Any, int]:
        with self.lock:
            if key in self.knowledge:
                return self.knowledge[key].value, self.knowledge[key].version
            return None, 0

    def update_if_latest(self, key: str, value: Any, expected_version: int) -> bool:
        with self.lock:
            if key in self.knowledge and self.knowledge[key].version == expected_version:
                self.knowledge[key] = VersionedKnowledge(value, expected_version + 1)
                return True
            return False

class ConsistentKnowledgeSharingAgent(Agent):
    def __init__(self, agent_id: str, knowledge_base: ConsistentKnowledgeBase):
        super().__init__(agent_id)
        self.knowledge_base = knowledge_base

    def share_knowledge(self, key: str, value: Any):
        version = self.knowledge_base.add_or_update_knowledge(key, value)
        print(f"Agent {self.agent_id} shared knowledge: {key} = {value} (version {version})")

    def retrieve_knowledge(self, key: str) -> Tuple[Any, int]:
        value, version = self.knowledge_base.get_knowledge(key)
        print(f"Agent {self.agent_id} retrieved knowledge: {key} = {value} (version {version})")
        return value, version

    def update_shared_knowledge(self, key: str, value: Any, expected_version: int) -> bool:
        success = self.knowledge_base.update_if_latest(key, value, expected_version)
        if success:
            print(f"Agent {self.agent_id} updated shared knowledge: {key} = {value} (new version {expected_version + 1})")
        else:
            print(f"Agent {self.agent_id} failed to update knowledge: {key} (expected version {expected_version})")
        return success

# 使用示例
consistent_kb = ConsistentKnowledgeBase()

agent1 = ConsistentKnowledgeSharingAgent("A1", consistent_kb)
agent2 = ConsistentKnowledgeSharingAgent("A2", consistent_kb)

agent1.share_knowledge("status", "normal")
_, version = agent2.retrieve_knowledge("status")
agent2.update_shared_knowledge("status", "alert", version)
agent1.retrieve_knowledge("status")

# 尝试使用旧版本更新，应该失败
agent1.update_shared_knowledge("status", "critical", version)
```

### 10.3.3 增量学习与知识传播

实现机制使 Agent 能够从新的经验中学习，并将这些新知识传播给其他 Agent。

示例（增量学习和知识传播系统）：

```python
from typing import List, Callable

class KnowledgeItem:
    def __init__(self, key: str, value: Any, confidence: float):
        self.key = key
        self.value = value
        self.confidence = confidence

class IncrementalLearningAgent(Agent):
    def __init__(self, agent_id: str, knowledge_base: ConsistentKnowledgeBase, learning_rate: float):
        super().__init__(agent_id)
        self.knowledge_base = knowledge_base
        self.learning_rate = learning_rate
        self.local_knowledge: Dict[str, KnowledgeItem] = {}

    def learn(self, key: str, value: Any, confidence: float):
        if key in self.local_knowledge:
            old_item = self.local_knowledge[key]
            new_confidence = old_item.confidence + self.learning_rate * (confidence - old_item.confidence)
            new_value = self.combine_knowledge(old_item.value, value, old_item.confidence, confidence)
        else:
            new_confidence =confidence
            new_value = value

        self.local_knowledge[key] = KnowledgeItem(key, new_value, new_confidence)
        print(f"Agent {self.agent_id} learned: {key} = {new_value} (confidence: {new_confidence:.2f})")

    def combine_knowledge(self, old_value: Any, new_value: Any, old_confidence: float, new_confidence: float) -> Any:
        # 这里应该实现具体的知识组合逻辑
        # 简单起见，我们只返回置信度更高的值
        return new_value if new_confidence > old_confidence else old_value

    def propagate_knowledge(self):
        for item in self.local_knowledge.values():
            current_value, version = self.knowledge_base.get_knowledge(item.key)
            if current_value is None or item.confidence > 0.8:  # 只传播高置信度的知识
                new_version = self.knowledge_base.add_or_update_knowledge(item.key, item.value)
                print(f"Agent {self.agent_id} propagated knowledge: {item.key} = {item.value} (version {new_version})")

    def receive_propagated_knowledge(self, key: str):
        value, _ = self.knowledge_base.get_knowledge(key)
        if value is not None and (key not in self.local_knowledge or self.local_knowledge[key].confidence < 0.9):
            self.learn(key, value, 0.7)  # 假设传播的知识有0.7的初始置信度

class KnowledgePropagationSystem:
    def __init__(self):
        self.knowledge_base = ConsistentKnowledgeBase()
        self.agents: List[IncrementalLearningAgent] = []

    def add_agent(self, agent: IncrementalLearningAgent):
        self.agents.append(agent)

    def global_learning_cycle(self):
        for agent in self.agents:
            agent.propagate_knowledge()
        
        for agent in self.agents:
            for key in self.knowledge_base.knowledge.keys():
                agent.receive_propagated_knowledge(key)

# 使用示例
propagation_system = KnowledgePropagationSystem()

agent1 = IncrementalLearningAgent("A1", propagation_system.knowledge_base, learning_rate=0.3)
agent2 = IncrementalLearningAgent("A2", propagation_system.knowledge_base, learning_rate=0.3)

propagation_system.add_agent(agent1)
propagation_system.add_agent(agent2)

# 模拟学习和知识传播
agent1.learn("temperature", 25, 0.8)
agent2.learn("temperature", 27, 0.7)
agent1.learn("humidity", 60, 0.9)

print("\nRunning global learning cycle:")
propagation_system.global_learning_cycle()

print("\nFinal knowledge state:")
for agent in propagation_system.agents:
    print(f"\nAgent {agent.agent_id} knowledge:")
    for key, item in agent.local_knowledge.items():
        print(f"- {key}: {item.value} (confidence: {item.confidence:.2f})")
```

## 10.4 集体决策机制

在多 Agent 系统中，集体决策机制允许 Agent 群体共同做出决策，这通常能产生比单个 Agent 更好的结果。

### 10.4.1 投票算法

实现一个简单的投票系统，允许 Agent 对不同选项进行投票。

```python
from collections import Counter
from typing import List, Dict

class VotingSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def collect_votes(self, options: List[str]) -> Dict[str, int]:
        votes = Counter()
        for agent in self.agents:
            vote = agent.cast_vote(options)
            votes[vote] += 1
        return dict(votes)

    def simple_majority_vote(self, options: List[str]) -> str:
        votes = self.collect_votes(options)
        winner = max(votes, key=votes.get)
        return winner

class VotingAgent(Agent):
    def cast_vote(self, options: List[str]) -> str:
        # 在实际应用中，这里应该实现更复杂的决策逻辑
        return random.choice(options)

# 使用示例
agents = [VotingAgent(f"A{i}") for i in range(10)]
voting_system = VotingSystem(agents)

options = ["Option A", "Option B", "Option C"]
result = voting_system.simple_majority_vote(options)

print(f"Voting result: {result}")
```

### 10.4.2 拍卖机制

实现一个简单的拍卖系统，用于资源分配或任务分配。

```python
class AuctionItem:
    def __init__(self, item_id: str, description: str, starting_bid: float):
        self.item_id = item_id
        self.description = description
        self.starting_bid = starting_bid
        self.current_bid = starting_bid
        self.highest_bidder = None

class AuctionSystem:
    def __init__(self):
        self.items: Dict[str, AuctionItem] = {}

    def add_item(self, item: AuctionItem):
        self.items[item.item_id] = item

    def place_bid(self, item_id: str, bidder: Agent, bid_amount: float) -> bool:
        if item_id not in self.items:
            return False
        item = self.items[item_id]
        if bid_amount > item.current_bid:
            item.current_bid = bid_amount
            item.highest_bidder = bidder
            return True
        return False

    def conclude_auction(self, item_id: str) -> Tuple[Agent, float]:
        if item_id not in self.items:
            return None, 0
        item = self.items[item_id]
        return item.highest_bidder, item.current_bid

class BiddingAgent(Agent):
    def __init__(self, agent_id: str, budget: float):
        super().__init__(agent_id)
        self.budget = budget

    def place_bid(self, item: AuctionItem, auction_system: AuctionSystem):
        if self.budget > item.current_bid:
            bid_amount = min(self.budget, item.current_bid * 1.1)  # 简单的出价策略
            if auction_system.place_bid(item.item_id, self, bid_amount):
                print(f"Agent {self.agent_id} placed a bid of {bid_amount} on {item.description}")
                return True
        return False

# 使用示例
auction_system = AuctionSystem()
item = AuctionItem("ITEM1", "Valuable Resource", 100.0)
auction_system.add_item(item)

agents = [BiddingAgent(f"A{i}", random.uniform(80, 200)) for i in range(5)]

for _ in range(3):  # 模拟3轮竞价
    for agent in agents:
        agent.place_bid(item, auction_system)

winner, final_price = auction_system.conclude_auction("ITEM1")
if winner:
    print(f"Auction concluded. Winner: Agent {winner.agent_id}, Price: {final_price}")
else:
    print("Auction concluded with no winner.")
```

### 10.4.3 共识算法

实现一个简化版的共识算法，允许 Agent 就某个值达成一致。

```python
import random
from typing import List, Dict

class ConsensusAgent(Agent):
    def __init__(self, agent_id: str, initial_value: float):
        super().__init__(agent_id)
        self.value = initial_value
        self.neighbors: List[ConsensusAgent] = []

    def add_neighbor(self, neighbor: 'ConsensusAgent'):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def update_value(self):
        if not self.neighbors:
            return
        
        neighbor_values = [neighbor.value for neighbor in self.neighbors]
        self.value = sum(neighbor_values) / len(neighbor_values)

class ConsensusSystem:
    def __init__(self, agents: List[ConsensusAgent]):
        self.agents = agents

    def run_consensus(self, max_iterations: int, tolerance: float) -> bool:
        for _ in range(max_iterations):
            old_values = [agent.value for agent in self.agents]
            
            for agent in self.agents:
                agent.update_value()
            
            new_values = [agent.value for agent in self.agents]
            
            if max(abs(new - old) for new, old in zip(new_values, old_values)) < tolerance:
                return True  # Consensus reached
        
        return False  # Max iterations reached without consensus

# 使用示例
agents = [ConsensusAgent(f"A{i}", random.uniform(0, 100)) for i in range(5)]

# 创建一个完全连接的网络
for i, agent in enumerate(agents):
    for j, neighbor in enumerate(agents):
        if i != j:
            agent.add_neighbor(neighbor)

consensus_system = ConsensusSystem(agents)

print("Initial values:")
for agent in agents:
    print(f"Agent {agent.agent_id}: {agent.value:.2f}")

consensus_reached = consensus_system.run_consensus(max_iterations=100, tolerance=0.01)

print("\nFinal values:")
for agent in agents:
    print(f"Agent {agent.agent_id}: {agent.value:.2f}")

print(f"\nConsensus {'reached' if consensus_reached else 'not reached'}")
```

这些示例展示了多 Agent 系统中的集体决策机制，包括投票、拍卖和共识算法。在实际应用中，这些机制可能需要更复杂的实现，考虑更多的因素，如 Agent 的个体特征、系统的动态性、通信延迟等。此外，还需要考虑安全性和公平性，以防止恶意 Agent 操纵决策过程。

## 10.5 多 Agent 学习

多 Agent 学习是一个复杂的领域，涉及多个 Agent 同时学习和适应环境以及彼此的行为。这种学习可以产生复杂的动态行为和创新性的解决方案。

### 10.5.1 协作强化学习

在协作强化学习中，多个 Agent 共同学习以最大化共同的奖励。

示例（简化的协作强化学习系统）：

```python
import numpy as np
from typing import List, Tuple

class Environment:
    def __init__(self, size: int):
        self.size = size
        self.state = np.zeros((size, size))
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.state.fill(0)
        self.state[self.goal] = 1
        return (0, 0)

    def step(self, action: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        new_position = (
            max(0, min(self.size - 1, action[0])),
            max(0, min(self.size - 1, action[1]))
        )
        reward = 1 if new_position == self.goal else -0.1
        done = new_position == self.goal
        return new_position, reward, done

class CollaborativeAgent:
    def __init__(self, agent_id: str, learning_rate: float, discount_factor: float, epsilon: float):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_action(self, state: Tuple[int, int], available_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        if np.random.random() < self.epsilon:
            return random.choice(available_actions)
        
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def get_q_value(self, state: Tuple[int, int], action: Tuple[int, int]) -> float:
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: Tuple[int, int], action: Tuple[int, int], reward: float, next_state: Tuple[int, int], next_actions: List[Tuple[int, int]]):
        current_q = self.get_q_value(state, action)
        next_max_q = max(self.get_q_value(next_state, next_action) for next_action in next_actions)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

class CollaborativeQLearning:
    def __init__(self, env: Environment, agents: List[CollaborativeAgent]):
        self.env = env
        self.agents = agents

    def train(self, episodes: int):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                actions = []
                for agent in self.agents:
                    available_actions = [(state[0] + dx, state[1] + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
                    action = agent.get_action(state, available_actions)
                    actions.append(action)

                # 选择一个 Agent 的动作执行
                chosen_action = random.choice(actions)
                next_state, reward, done = self.env.step(chosen_action)
                total_reward += reward

                # 所有 Agent 更新 Q 值
                next_available_actions = [(next_state[0] + dx, next_state[1] + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
                for agent in self.agents:
                    agent.update_q_value(state, chosen_action, reward, next_state, next_available_actions)

                state = next_state

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

# 使用示例
env = Environment(5)
agents = [
    CollaborativeAgent("A1", learning_rate=0.1, discount_factor=0.95, epsilon=0.1),
    CollaborativeAgent("A2", learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
]

collaborative_q_learning = CollaborativeQLearning(env, agents)
collaborative_q_learning.train(1000)

# 测试学习结果
state = env.reset()
done = False
while not done:
    actions = [agent.get_action(state, [(state[0] + dx, state[1] + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]) for agent in agents]
    chosen_action = random.choice(actions)
    state, reward, done = env.step(chosen_action)
    print(f"State: {state}, Action: {chosen_action}, Reward: {reward}")
```

### 10.5.2 对抗性学习

在对抗性学习中，Agent 学习在竞争环境中表现。这可以通过实现一个简化的双人零和游戏来演示。

示例（简化的对抗性学习系统）：

```python
import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        row, col = action
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True, {}  # Invalid move

        self.board[row, col] = self.current_player
        done = self.check_win() or np.all(self.board != 0)
        reward = 1 if self.check_win() else 0 if done else 0
        self.current_player = 3 - self.current_player  # Switch player (1 -> 2, 2 -> 1)
        return self.board.copy(), reward, done, {}

    def check_win(self):
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                return True
        if np.all(np.diag(self.board) == self.current_player) or np.all(np.diag(np.fliplr(self.board)) == self.current_player):
            return True
        return False

class AdversarialAgent:
    def __init__(self, player, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.player = player
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_action(self, state):
        state_key = self.state_to_key(state)
        if np.random.random() < self.epsilon:
            return self.random_action(state)
        return self.greedy_action(state_key)

    def random_action(self, state):
        available_actions = [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
        return random.choice(available_actions)

    def greedy_action(self, state_key):
        if state_key not in self.q_table:
            return self.random_action(self.key_to_state(state_key))
        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in self.possible_actions(state)}
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0 for action in self.possible_actions(next_state)}
        
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def state_to_key(self, state):
        return tuple(state.flatten())

    def key_to_state(self, key):
        return np.array(key).reshape((3, 3))

    def possible_actions(self, state):
        return [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]

def train_agents(env, agent1, agent2, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if env.current_player == 1:
                action = agent1.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent1.update_q_value(state, action, reward, next_state)
            else:
                action = agent2.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent2.update_q_value(state, action, -reward, next_state)  # Note the negative reward
            
            state = next_state
        
        if episode % 1000 == 0:
            print(f"Episode {episode} completed")

# 使用示例
env = TicTacToeEnv()
agent1 = AdversarialAgent(player=1)
agent2 = AdversarialAgent(player=2)

train_agents(env, agent1, agent2, 10000)

# 测试学习结果
state = env.reset()
done = False
while not done:
    if env.current_player == 1:
        action = agent1.get_action(state)
    else:
        action = agent2.get_action(state)
    state, reward, done, _ = env.step(action)
    print(f"Player {env.current_player} action: {action}")
    print(state)
    print()

if reward == 1:
    print(f"Player {3 - env.current_player} wins!")
elif reward == 0:
    print("It's a draw!")
```

### 10.5.3 元学习在多 Agent 系统中的应用

元学习允许 Agent 学习如何更有效地学习。在多 Agent 系统中，这可以帮助 Agent 快速适应新的任务或环境。

示例（简化的多 Agent 元学习系统）：

```python
import numpy as np
from typing import List, Tuple

class Task:
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)

    def get_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.randn(self.input_dim)
        y = np.dot(x, self.weights)
        return x, y

class MetaLearningAgent:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.w1))
        return np.dot(h, self.w2)

    def update(self, x: np.ndarray, y: np.ndarray):
        # Simplified update rule (not a true backpropagation)
        h = np.tanh(np.dot(x, self.w1))
        y_pred = np.dot(h, self.w2)
        error = y - y_pred

        self.w2 += self.learning_rate * np.outer(h, error)
        self.w1 += self.learning_rate * np.outer(x, np.dot(error, self.w2.T) * (1 - h**2))

class MultiAgentMetaLearningSystem:
    def __init__(self, num_agents: int, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float):
        self.agents = [MetaLearningAgent(input_dim, hidden_dim, output_dim, learning_rate) for _ in range(num_agents)]

    def meta_train(self, num_tasks: int, num_episodes: int, num_steps: int):
        for _ in range(num_episodes):
            task = Task(self.agents[0].input_dim, self.agents[0].output_dim)
            
            for agent in self.agents:
                for _ in range(num_steps):
                    x, y = task.get_sample()
                    y_pred = agent.forward(x)
                    agent.update(x, y)

    def evaluate(self, task: Task, num_steps: int) -> List[float]:
        errors = []
        for agent in self.agents:
            agent_errors = []
            for _ in range(num_steps):
                x, y = task.get_sample()
                y_pred = agent.forward(x)
                error = np.mean((y - y_pred)**2)
                agent_errors.append(error)
                agent.update(x, y)
            errors.append(agent_errors)
        return errors

# 使用示例
input_dim = 5
hidden_dim = 10
output_dim = 3
learning_rate = 0.01
num_agents = 3

meta_learning_system = MultiAgentMetaLearningSystem(num_agents, input_dim, hidden_dim, output_dim, learning_rate)

# 元训练
meta_learning_system.meta_train(num_tasks=100, num_episodes=1000, num_steps=10)

# 评估
evaluation_task = Task(input_dim, output_dim)
errors = meta_learning_system.evaluate(evaluation_task, num_steps=50)
```

