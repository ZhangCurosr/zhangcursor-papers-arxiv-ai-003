# Planarian: Managing Agent State with Statepoints

Jinnan Guo Imperial College London

Hao (Mark) Chen Imperial College London

Kapil Vaswani Indian Institute of Science Bangalore SPARC

Andrew Paverd Microsoft Security Response Center

Peter Pietzuch Imperial College London

## Abstract

LLM agents solve complex tasks by iteratively changing files, invoking local tools, and interacting with remote services, which modifies state across their local environment and remote services. Today, agents and users must manage these changes explicitly, whether reverting exploratory actions or recovering from erroneous ones. Doing so safely requires coordinated actions, yet current agent harnesses lack unified abstractions and mechanisms for managing local and remote state consistently and eficiently.

We describe Planarian, an agent runtime with state management that enables agents and users to recover from erroneous actions and explore alternative executions over consistent local and remote environment state. Planarian introduces the abstraction of agent statepoints, which are consistent, restorable point-in-time versions of the environment state. Planarian exposes three state-management primitives to agents and users: (i) snapshot creates a new statepoint spanning local and remote state without requiring external services to support checkpoints: it relies on eficient incremental process and file system snapshotting to capture local sandboxed state, and transparently records compensating actions to undo remote state changes; (ii) rollback restores the environment to a previous statepoint by reverting to a prior local checkpoint and replaying compensating actions for remote state changes; and (iii) fork creates multiple isolated branches from a statepoint, enabling the agent to explore alternatives in parallel. We show that Planarian enables agents to undo mistakes and explore alternatives in parallel, improving task quality by up to 15×, and allows users to recover from erroneous actions with only 3% overhead.

## 1 Introduction

Agents perform tasks that require repeated interaction with a host machine, from modifying software to administering services [3, 4, 15]. An agent harness invokes a large language model (LLM) with context comprising instructions, prior actions, and tool results. It uses the model’s output to invoke tools and incorporates their results into subsequent context. Tool calls request commands or service operations that act on environment state: local files and processes, and state held by external services, such as database state. Local tools can execute directly on the host or within a sandbox that isolates their execution. The harness can invoke remote tools through the model context protocol (MCP) [2].

When an agent fails to complete a task, it may have to undo earlier actions before trying diferent steps. A user may also want to reverse an unwanted change, such as the deletion of files or incorrect updates to a remote calendar service. For the agent, reconstructing state requires extra tool calls, which may be costly or impossible if data was lost. We observe that explicit state management would let agents and users recover from mistakes without manually having to repair the afected state. Agents could also explore the efect of alternative actions on the environment eficiently without repeating successful work or carrying over unwanted efects.

Supporting such state management alongside existing agent harnesses, however, raises three challenges:

(1) Consistency across local and remote state. State management requires environment consistency, i.e., it must restore both local and remote state consistently. Local files and processes can be checkpointed, but MCP endpoints of remote network services may provide neither snapshots nor undo operations. Recovering only local state can result in inconsistencies though: for example, a data-cleaning agent may update database records through an MCP endpoint and write a local report of its changes. Reverting only the local report leaves it inconsistent with the updated remote database.

(2) Integration with agent execution. Users and agents must be able to perform state management through existing agent harnesses. In addition, state management must guarantee context consistency, i.e., the agent must be aware of the changes to the environment state. After the environment state is restored, the history of tool calls may describe efects that no longer exist. Simply retaining this history can mislead the agent; discarding it loses evidence of unsuccessful actions that could guide future actions. State management therefore afects both how the harness dispatches tools and what context it supplies.

(3) Overhead of state management. Frequent state management must not dominate task execution time or storage use. Local state, including files and process memory, may be large, even when individual tool calls change little state. State management overheads may accumulate both as execution advances and whenever an earlier state is revisited to try diferent actions. Managing remote state also incurs communication latency and is constrained by MCP interfaces.

Prior approaches solve only some of these challenges. Checkpointing systems such as DMTCP [18] and Catalyzer [25] preserve local execution state, but cannot account for remote ser vice state or agent decisions. Transactional workflows [45], speculative configuration repair [37], and intrusion recovery with external compensation [30] establish broader recovery models, but target generic workflows or application repair rather than agent exploration. Without environment and context consistency, state management cannot serve the par ties that need it: an agent cannot safely roll back on its own, a harness cannot search over alternative branches, and a user cannot reliably undo an unwanted change. The open problem is thus to make an agent’s evolving environment state a unit of recovery and exploration, which is consistent across local and remote state, consistent with the agent’s context, and eficient enough to be used repeatedly.

We describe Planarian, an agent runtime for state management that operates together with existing agent harnesses. Planarian provides a unified abstraction for managing local and remote environment state, enabling consistent recovery and exploration while supplying the information needed for subsequent agent decisions. For this, Planarian makes three technical contributions:

(1) Agent statepoints and primitives. Recovery and exploration need a common abstraction for managing both local and remote state. Planarian introduces agent statepoints: a statepoint captures both local files and running processes together with the remote state changed through MCP operations. A statepoint remains restorable even when the remote service cannot take checkpoints, and carries a description of the captured state and the outcomes of prior executions. For state management, three primitives operate over statepoints: (i) snapshot records a new statepoint; (ii) rollback returns to a statepoint to undo state changes; and (iii) fork starts an isolated alternative execution in a branch from a statepoint, while preserving the existing state.

(2) Integrating agent statepoints with agent execution. Planarian exposes its state-management primitives (a) as tools that agents invoke directly, or through APIs for (b) harness driven search or (c) manual user requests. During execution, the agent must select suitable statepoints and interpret them correctly. Planarian uses the evidence associated with statepoints both to inform this choice and to update the agent context afterward. The agent harness retains responsibility for LLM invocations and context construction, routing tool calls through Planarian and incorporating the returned results. Planarian manages environment state through local tool sandboxing and MCP proxying, and captures statepoints between tool calls. After restoration, Planarian supplies the selected state’s description and prior outcomes to the agent harness to append to the context. This identifies the environment state now available to agents without discarding the information needed to choose diferent actions.

![](images/9feddd6de829c9abe73692fbe9f0723c662e2594993e675132c8ba587de2c7e3.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 1<sub>.</sub> A<sub>ge</sub>nt h<sub>a</sub>rn<sub>ess</sub> inv<sub>o</sub>kin<sub>g</sub> th<sub>e</sub> LLM <sub>a</sub>nd t<sub>oo</sub>l<sub>s</sub>

(3) Local checkpointing and remote compensation. To keep the overhead of state management low, Planarian limits the state that must be copied locally and reconstructed remotely when creating or reverting to statepoints. For local state, Planarian combines incremental copy-on-write filesystem snapshots [17] with CRIU process checkpoints [7] that record only changed memory. Planarian re-establishes the restored checkpoint as the baseline for subsequent memory changes, allowing incremental capture across rollback and fork without modifying CRIU or the kernel. This avoids repeatedly copying unchanged memory during exploration. For remote state, Planarian proxies supported MCP requests so that their efects can be reversed through compensating actions—service operations that undo earlier changes. For example, a database MCP proxy preserves original database row values and afected keys within the transaction that modifies them, allowing destructive updates to be undone. We evaluate Planarian with agents in coding, personal assistance, system administration, database management, and gaming tasks. In the Mario platform game, an agent that uses Planarian to retry from statepoints achieves 15× the game score of an agent without state management. When solving system-administration tasks with MCTS-style tree search, Planarian takes snapshots 10× faster than a baseline that uses full process and filesystem checkpoints. Across system and database administration tasks, state management adds only 1%–3% to completion time, including coordinated recovery for an agent that modifies a remote database through MCP. These results show that Planarian enables agents to revisit decisions and explore alternatives while incurring little overhead for state management.

## 2 State Management for Agents

First we describe agent execution (§2.1), establish the requirements for managing environment state (§2.2), and assess existing state-management approaches (§2.3).

## 2.1 Agent execution

As LLMs become capable of using tools, users delegate everyday tasks, such as coding, system administration and data maintenance, to agents for automation [33, 39].

As shown in Fig. 1, an agent uses an LLM to choose actions toward a task objective. An agent harness coordinates the model and tool invocations through which the agent interacts with this environment through an agent loop [43]:

it invokes the LLM with context, interprets the output to invoke tools, and incorporates tool results into the context for subsequent LLM invocations. The agent’s environment state thus comprises local machine state, such as files and running processes, and external network service state.

Agent harness implementations range from general ones (e.g., LangChain [5] and AutoGen [41]) to specialized harnesses for coding agents (e.g., Claude Code [3], Codex [4]) and personal assistants (e.g., OpenClaw [15], Nanobot [13]).

Tools provide access to local commands and files and remote service operations. An agent harness can invoke local tools on the host and remote tools through the model context protocol (MCP) [2]. MCP provides a common interface for invoking APIs exposed by remote servers.

Sandboxing [36] is an optional safety feature that confines local tool execution to an isolated environment. Restricting access to host files and processes limits the damage that erroneous commands can cause outside the sandbox. Sandboxes can be realized using containers [31], which isolate processes and their filesystem view through OS mechanisms, such as Linux namespaces. A harness creates and starts a container from an image containing the required software, executes local tool calls inside it, and removes it when the task ends. Sandboxing libraries and SDKs, such as E2B [12], provide APIs through which harnesses can use sandboxing.

## 2.2 Managing environment state

As an agent executes, it mutates the environment state, producing visible side efects on the machine. When an agent updates environment state, there are two broad requirements: Efectiveness. Agents often solve challenging tasks through search and exploration [44]. By sampling multiple rollouts from an LLM, an agent can improve its problem-solving capability by allowing it to try alternative plans and learn from unsuccessful attempts. Such exploration mechanisms are increasingly common: pass@� allows an agent to attempt a task � times [22]; Monte Carlo tree search (MCTS) [38] performs exploration and backtracking as part of the harness; and EAPO [27] encourages the agent to explore and recover from mistakes through backtracking.

At the system level, an agent must be able to explore plans sequentially or concurrently, with some inevitably failing. These approaches impact the environment state, as they require backtracking and state branching so that side efects of old plans do not afect subsequent execution.

Safety. Since LLM outputs are non-deterministic and unreliable (cf. hallucination [28], sycophancy [26]), an agent’s tool calls can produce unintended, harmful side efects, e.g., deleting required files, installing incorrect packages, or committing incorrect writes to a database. Unlike a conventional program, an agent writes and executes commands without human review, so it may break a production environment before the mistake is noticed by the user. Running an agent safely thus requires the ability to rollback the state after a mistake, preserving the integrity of the environment state.

We observe that supporting both efectiveness and safety requires state management that enables exploration of alternative executions and recovery from unwanted changes: exploration allows alternative actions to be tried without their efects interfering, while recovery undoes unwanted efects. State management must cover local and remote state, and maintain consistency across these types ofstate and with the agent’s context. Furthermore, recovery and exploration should be accessible to users, harnesses, and agents: a user may notice a mistake, a harness may organize a search, or an agent may decide to retry an action. Since each acts on the same environment, the guarantees must hold regardless of who initiates the operation. These state management capabilities must also be implemented eficiently to not degrade the performance of agent execution.

## 2.3 Existing state management approaches

Prior approaches only support some of these requirements for state management. Tab. 1 compares approaches along four dimensions: the environment state columns describe whether local files, processes and remote service state are managed; the consistency column refers to coordination between local and remote state (L/R), and between agent context and environment state (C/E); the primitives column considers support for recording, restoring, and branching versions of the environment state; and the stakeholders column identifies whether users, harness code, or agents can invoke state management.

Framework checkpointing. Systems such as LangChain [5] and OpenClaw [15] ofer framework checkpoints that preserve conversation history, intermediate results, and workflow progress. In terms of environment state, this covers at most parts of the local state and excludes remote state. Their primitives resume or revisit workflow steps, rather than restoring and branching the complete environment. Consequently, consistency between context and environment is limited: rewinding a conversation does not undo a database update, and restoring workspace files does not restore the processes using them. For stakeholders, these approaches support user- or harness-initiated recovery: orchestration code selects a saved workflow state and resumes execution from it. This does not itself provide agent-driven recovery, which would require exposing recovery choices to the LLM and interpreting its selection within the running agent loop. Sandbox checkpointing. Approaches such as Podman [16], E2B [12], CubeSandbox [8], and DeltaBox [24] provide sandbox checkpoints. Their environment state includes local files and processes, and their primitives support snapshot, rollback, and branching through user and harness interfaces. Some also preserve context with local state, providing C/E consistency, and recent systems reduce checkpointing costs for repeated exploration. However, remote service state remains outside sandboxes, so these guarantees do not extend to L/R consistency. A restored local report can thus disagree with database updates that remain committed. The fundamental limitation is state coverage: eficient local recovery does not undo remote efects. In terms of stakeholders, the compared interfaces let users and harnesses request checkpoints and restores. Agent-selected recovery would require connecting checkpoint contents to task progress so that the agent can choose where to resume.

Table 1. Existing approaches to agentic state management (local: file and process state; remote: state of external services; L/R: consistency between local and remote state; C/E: consistency between agentic context and environment state; ◦ means partial support.)
<table><tr><td rowspan="2">Class</td><td rowspan="2">Approach</td><td colspan="2">Environment state</td><td colspan="2">Consistency</td><td colspan="3">Primitives</td><td colspan="3">Stakeholders</td></tr><tr><td>Local</td><td>Remote</td><td>L/R</td><td>C/E</td><td>Snapshot</td><td>Rollback</td><td>Fork</td><td>User</td><td>Harness</td><td>Agent</td></tr><tr><td>Framework</td><td>LangChain [5]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>checkpoint</td><td>OpenClaw [15]</td><td>o</td><td>x</td><td>x</td><td>o</td><td>o</td><td>o</td><td>x</td><td>√</td><td>o</td><td>x</td></tr><tr><td rowspan="4">Sandbox</td><td>Podman [16]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>E2B [12]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>CubeSandbox [8]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>DeltaBox [24]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td rowspan="4">Workflow recovery</td><td>SagaLLM [21]</td><td>x</td><td>√</td><td>x</td><td>√</td><td>x</td><td>o</td><td>x</td><td>√</td><td>√</td><td>o</td></tr><tr><td>LogAct [20]</td><td>o</td><td>√</td><td>o</td><td>√</td><td>x</td><td>o</td><td>x</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Cordon [23]</td><td>o</td><td>0</td><td>o</td><td>x</td><td>o</td><td>o</td><td>x</td><td>√</td><td>√</td><td>x</td></tr><tr><td>PLANARIAN</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Workflow recovery. Approaches such as SagaLLM [21], LogAct [20], and Cordon [23] address tool efects through corrective actions or delayed commitment. Their environment state can include remote efects and selected local changes, but excludes running process state. In history-based approaches, histories and dependency records support C/E consistency and guide recovery by harnesses or agents; L/R consistency remains limited to the efects covered by their recovery mechanisms. Their primitives consequently have narrower semantics than restoring and branching arbitrary earlier environments. Compensation requires suitable undo operations and retained recovery information; staging supports abort before commitment, but cannot undo efects already released. Whether agent-driven primitives are supported depends on who controls recovery: approaches that expose execution histories to the LLM can ask it to diagnose failures and choose corrective actions; transaction-based approaches carry out commit/abort decisions outside the agent’s action-selection loop.

None of these prior approaches achieve all four dimensions. As summarized in Tab. 1, Planarian combines these capabilities: it coordinates local and remote restoration, reflects the restored environment in agent context, and exposes the same state-management primitives to users, harnesses, and agents through a new unified state management abstraction.

## 3 Agent Statepoints

In this section, we introduce agent statepoints, a new abstraction that unifies the management of an agent’s environment

![](images/a838545915be900114480e8095f137c78528e9c8cbb2da5d2d7299d32e4488e1.jpg)  
Fi<sub>g</sub>ure 2<sub>.</sub> Overview of a<sub>g</sub>ent state<sub>p</sub>oints

state. We present the statepoint concept (§3.1), and describe the primitives that it enables (§3.2).

## 3.1 Concept

As Fig. 2 shows, an agent statepoint is a consistent, restorable point-in-time version of an agent’s environment state, together with evidence that describes it. It provides a common object through which users, harnesses, and agents can reason about exploration and recovery. An agent action can change both local and remote state, so a statepoint groups them into a single version that can be recovered consistently.

A statepoint spans two state domains: the local state comprises the files and running processes within the agent’s sandbox. Statepoints require local tool execution to be confined to a sandbox: its boundary defines which files and processes must be captured and restored together, while isolating them from unrelated host state. The remote state comprises external service state afected by the agent’s supported operations, such as database updates through MCP requests. Unlike the local state, the remote state is owned by the service provider instead of the agent, so the agent can observe and mutate this state only through the service API. The statepoint represents a recoverable version of this remote state without requiring the service to provide a checkpoint.

The agent statepoint must achieve cross-state consistency: its local and remote state must reflect the agent-observable state at a single point in time. Otherwise, independently captured local and remote state could describe an environment that never existed. Treating them as one version makes consistency a requirement of the abstraction, rather than a responsibility of each caller.

![](images/e972572066b50c194dd373e550ef035dea6fece35fa67efcb853b58fee0eb7c1.jpg)  
Fi<sub>g</sub>ure 3<sub>.</sub> State mana<sub>g</sub>ement <sub>p</sub>rimitives for a<sub>g</sub>ent state<sub>p</sub>oints

Each statepoint also carries state evidence that gives the environment version meaning: a description of its state and the outcomes of executions that previously proceeded from it. The evidence helps stakeholders (e.g., the agent or user) select an appropriate statepoint to restore and avoid repeating executions that already failed. Evidence is associated metadata, distinct from both the captured environment state and the agent’s LLM context. The evidence can be extended with a summary of what an execution attempted after that statepoint and what happened, such as completing the task or encountering an error. Adding this information leaves the recorded environment version unchanged. Keeping the evidence separate from the environment state allows knowledge of unsuccessful attempts to guide subsequent decisions even after returning to an earlier version.

## 3.2 Primitives

The agent statepoint enables three state-management primitives (see Fig. 3): snapshot, rollback, and fork, which meet the recovery and exploration requirements in §2.2. They operate on consistent environment versions and their associated state evidence, giving users, harnesses, and agents the same guarantees without requiring each caller to coordinate local and remote state.

Snapshot. To return to an earlier execution point requires that the state was preserved consistently. Snapshot captures both local and remote state and generates a new statepoint. It records the environment at a chosen boundary between tool calls as one restorable version, together with a state description. In Fig. 3, each transition from statepoint 1 to 2 and from 2 to 3 comprises one or more tool calls followed by an explicit snapshot; note that the abstraction does not require a snapshot after every call. Each statepoint covers local files and running processes as well as the afected remote service state, even when the service provides no checkpoint API.

Rollback. Recovering from an error or abandoning an unsuccessful attempt requires undoing its efects across both state domains. Rollback restores the environment state to a previously snapshotted statepoint. In Fig. 3, returning from statepoint 3 to 2 abandons the later version shown in gray. For example, an agent records statepoint 2 after installing dependencies, then creates a file and registers it in a database. Rollback must undo both the file creation and the database update while preserving the dependencies. The state evidence helps callers choose where to return and identifies the restored environment. The agent’s context is not rewound, and it retains knowledge of unsuccessful attempts [46].

![](images/14fea628b6a929c6402cc4ef0584e896d1626ca7d6e2ee342fafc74d0a627589.jpg)  
Fi<sub>g</sub>ure 4. Desi<sub>g</sub>n of Planarian

Fork. If the agent explores, the environment state of each attempt must remain isolated. Fork creates a new environment branch from a previously captured statepoint. In Fig. 3, statepoint 2<sup>′</sup> branches from statepoint 2; alternative tool calls followed by a snapshot produce statepoint 3<sup>′</sup>. Fork preserves the source branch, allowing it to continue independently. Local files and processes are isolated between branches; extending this isolation to remote state requires the service to support independent branches. The associated state evidence describes the common starting point and prior attempts, helping callers choose alternative actions.

## 4 Planarian Design

Planarian is an agent runtime designed to realize the agent statepoint abstraction and state management primitives in §3. In this section, we describe the overall design of Planarian, and explain how it enables state management for agents. After that, §5 and §6 provide specific implementation details of how Planarian manages local and remote state, respectively.

## 4.1 Overview

Fig. 4 shows Planarian’s architectural components. During agentic execution, the state management functionality provided by Planarian can be invoked by the user, the harness, or the agent itself. The harness routes tool execution through Planarian and integrates with Planarian via a context library. Tab. 2 shows the state-management primitives Planarian provides. A caller uses snapshot to record a statepoint, rollback to return the sandbox and remote state to that version, or the local fork operation to create another sandbox from its local state. The sandbox handle sb identifies the local execution environment, and sp identifies a statepoint, whose local and remote parts of state are sp-local and sp-remote. The capture and compensate hooks connect remote-service recovery to snapshot and rollback; the statepoint manager coordinates them with the sandbox operations.

## 4.2 Local state management

To enable state management, all tool invocations that read or mutate local state (e.g., processes, filesystems, etc.) must be run within a sandbox. This is already considered best practice for agentic systems in order to isolate the agent’s execution from the host environment.

In Planarian, a sandbox manager creates and manages one or more sandboxes and dispatches the agent’s tool calls to the appropriate sandbox. When a user, harness, or agent requests a snapshot, the sandbox manager captures the local state at a boundary between tool invocations. Callers can request a snapshot after each call or after several calls; the primitive does not prescribe a fixed frequency.

Delegating snapshot execution to the sandbox manager separates this choice from the mechanics of managing sandboxes and capturing their state. Planarian uses Podmanbased sandboxes [16], which combine incremental process checkpointing with zero-copy filesystem snapshots to reduce latency and copying overhead (§5). Snapshot and fork can also overlap LLM inference (§6.3), which can hide latency (§7.2).

## 4.3 Remote state management

For remote services without native versioning, Planarian relies on compensating actions that undo supported operations without requiring a service checkpoint. To achieve this, all tool calls that mutate remote state are routed through the state proxy—a transparent interception layer between the agent harness and the target MCP server (§6.1). For each supported tool call, the state proxy transforms the original request into a compensable form and records the information used to generate compensating actions. Requests that cannot be made compensable are rejected before execution.

When a statepoint is created, the state proxy records its current position in the sequence of MCP calls. On a rollback(sp, sb, compensate) request, it uses the recorded MCP call position to identify the subsequent actions and applies their compensating actions in reverse order.

## 4.4 State consistency

To fully realize the statepoint abstraction (contribution #1), Planarian must ensure that each statepoint is a consistent view of both the agent’s local and remote state. This is handled by the statepoint manager component, which coordinates between the sandbox manager and state proxy components. To create a new statepoint, it first freezes the sandbox, and takes the sandbox snapshot. After that, the MCP call position is captured from the state proxy. Once both parts are complete, the sandbox can be resumed.

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> P<sub>lanarian</sub>’<sub>s</sub> <sub>s</sub>t<sub>a</sub>t<sub>e</sub> <sub>managemen</sub>t API<sub>s</sub>
<table><tr><td>Function</td><td>Description</td></tr><tr><td>snapshot(sb,capture) → sp rollback(sp,sb,compensate)  $\mathsf { f o r k } ( \mathsf { s p } \mathrm { - } \mathsf { l o c a l } , \mathsf { s b } ) \to \mathsf { s b } ^ { \prime }$ </td><td>Records statepoint sp Restores sandbox, service Forks sandbox from</td></tr><tr><td>capture(sb,service) → sp-remote compensate(sp-remote,sb,service)</td><td>sp-local Hook: records remote state Hook: restores service</td></tr></table>

During a snapshot, the statepoint manager holds an exclusive lock: it waits for any in-flight commands and MCP calls to finish before commencing the snapshot, and new calls are blocked until the snapshot completes (§6.3). The local and remote state are combined into a single statepoint, representing a point-in-time version of the agent’s environment state, which is committed to the statepoint registry (§6.2).

Rolling back to a statepoint is also a two-step process: (i) the statepoint manager applies the compensating actions through the state proxy, bringing the remote state to its logically equivalent point; and (ii) it restores the local state of the sandbox from the snapshot.

The local fork(sp-local, sb) operation creates an isolated sandbox from a statepoint’s local state, preserving the original sandbox. The local checkpointing mechanisms in §5 support subsequent snapshots and rollbacks within each branch. Extending a branch to remote state requires serviceside branching, e.g., from a database such as Neon [14] or Dolt [11]. When a service supports compensation only, the forked sandbox cannot use that state proxy, preventing interference with the source branch (§6.2).

## 4.5 Agent context management

To integrate the statepoint primitives into the agent’s execution (contribution #2), the necessary information must be brought into the agent’s context. In Planarian, a context library collects state evidence and supplies information about state-management operations to the harness. It makes evidence available to users, harnesses, and agents when choosing a statepoint for rollback, and supplies the selected state’s description and prior outcomes after rollback. The harness retains responsibility for LLM invocation and context construction, including incorporating this information into the agent’s context. It exposes several APIs to the harness, as shown in Tab. 3:

Snapshot creation. The context library gathers state from the harness, e.g., the progress of a task, the player information from a game, or the state of the sandbox. This constructs a mapping between a statepoint and its state, making each statepoint self-describing. However, a static view of the agent’s state at the time of statepoint creation would be insuficient. For example, the state at that point may appear to be suitable for restoration, but it might lead to a deadlock or an undesirable outcome if restored. Therefore, Planarian also appends the execution outcomes that follow this statepoint, recording what previous executions did after this point. The description and the outcomes form the state evidence (§3.1), which guides the choice of statepoint for rollback. Fig. 5 shows the data recorded for each statepoint. Statepoint selection. Before a caller chooses a statepoint for rollback, the context library exposes the state ledger: the evidence associated with committed statepoints in the registry. Users and harnesses can inspect this evidence, and the harness can include it in the LLM’s context so that the agent can choose a suitable statepoint. Once rollback completes, the context library supplies the harness with the selected state’s description and prior execution outcomes, together with notification that rollback has taken place. This information forms the restore context returned by RestoreContext, which the harness appends to the context. By design, this operation does not remove existing context, so as not to interfere with the harness’s own context management, e.g., context compaction and memory. Without the restore context, the agent might make decisions based on outdated or inconsistent context, leading to repeated mistakes.

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> P<sub>lanarian</sub>’<sub>s con</sub>t<sub>ex</sub>t <sub>managemen</sub>t API<sub>s</sub>
<table><tr><td>Action</td><td>Context API</td><td>Description</td></tr><tr><td>snapshot</td><td>StateRecord(...)</td><td>Records state description Appends outcomes</td></tr><tr><td></td><td>AppendOutcome(...)</td><td></td></tr><tr><td>selection</td><td>ConstructLedger(...)</td><td>Exposes state evidence</td></tr><tr><td>rollback</td><td> $\mathsf { R e s t o r e c o n t e x t ( \ldots ) }$ </td><td>Supplies rollback context</td></tr></table>

## 5 Local state management

Planarian is implemented as a harness-agnostic runtime that integrates with agent harnesses by routing tool calls through the Planarian APIs. It uses Podman as the agent sandbox (§5.1), where the process tree is checkpointed using CRIU and the file system is snapshotted via ZFS (§5.2). Planarian introduces a process checkpoint protocol that keeps CRIU’s checkpoints incremental across restores (§5.3).

## 5.1 Sandbox management

Planarian provides a harness-agnostic sandbox manager that manages the lifecycle of the agent sandbox, including launching sandboxes, executing commands, and performing snapshot, rollback, and fork operations. For agents that do not have a sandboxed execution environment, we integrate Podman as the sandbox backend, since it is a securityhardened OCI container engine with first-class checkpoint support. The sandbox manager launches a long-running Podman container with its built-in isolation primitives, e.g., namespaces, cgroups, system call filtering, and access control, ensuring that an untrusted or erroneous agentic command cannot escape from the sandbox. Planarian uses the Podman container as the boundary of the local state. Each container’s root file system is mounted to a Planarian-managed ZFS [17] dataset, which is a block-level copy-on-write (CoW) filesystem.

## 5.2 Sandbox snapshot and the local state

The local state of a statepoint contains two parts: a process checkpoint and a file system snapshot. The container’s processes form a process tree rooted at the container’s PID 1. Planarian uses CRIU [7], a process checkpoint tool, to checkpoint the agent container’s process tree into an artifact. This enables Planarian to restore all application processes running at the time of the snapshot. Planarian also snapshots the container’s root file system, which on ZFS is a zero-copy, metadata-only operation. Together, they form a complete restore point of the sandbox. Using this mechanism, Planarian can roll back the file system and the applications running inside it to a clean state, making the sandbox recoverable by preserving its state rather than reconstructing it.

## 5.3 Continuous incremental checkpointing

To minimize the latency and storage cost of snapshots, Pla narian attempts to capture only the incremental changes since the last snapshot. CRIU already supports incremental checkpoints, but with one notable limitation: the first checkpoint after a restore must always be a full (i.e., nonincremental) checkpoint. There are two technical reasons for this: (1) CRIU’s incremental checkpointing relies on the kernel’s soft-dirty bits. After restoring from a checkpoint, the process will have a new address space and new page table entries, thus losing incremental memory tracking (i.e., all the soft-dirty bits will be set). (2) CRIU has an internal safety mechanism based on checkpoint creation time. This mechanism is designed to prevent accidentally creating an incremental checkpoint based on a parent checkpoint from a diferent process, even if the process ID (PID) is the same (e.g., through PID reuse).<sup>1</sup> Specifically, CRIU falls back to a full dump of a process whose start time is not earlier than the parent checkpoint’s creation time, which prevents incremental checkpoints after a restore.

Since agentic workloads perform very frequent snapshot and restore operations during exploration, creating a full checkpoint after each restore becomes very costly. Inspired by prior discussions in [1], Planarian proposes an incremental checkpoint protocol that requires no changes to the existing checkpoint/restore (C/R) stack (i.e., CRIU), but enables eficient incremental CRIU checkpointing across restores. The full protocol is shown in Algorithm 1.

Firstly, Planarian uses the insight that a freshly restored process should not have any of the dirty bits set because it has been reconstructed directly from a valid checkpoint. Planarian therefore uses CRIU’s pre-resume hook (line 9), which runs after the process tree has been restored but before execution resumes, to clear the soft-dirty bits of the restored processes (line 12).

Secondly, since Planarian can guarantee that the restored process is always a continuation of the checkpointed process, it can safely bypass CRIU’s checkpoint creation time safety mechanism. Specifically, in the CRIU pre-resume hook (line 14), Planarian computes the epoch time � as one tick after the latest start time of all restored threads. It then waits until the system uptime reaches or exceeds � (line 15), making � a boundary between the restored processes and any process created afterward. After the restore has completed, Planarian modifies the creation time of the parent checkpoint to be the epoch time � (i.e., so that it appears the parent checkpoint was created after the current process had started). This enables only the restored process to safely pass the checkpoint creation time check and create incremental checkpoints after the restore. If the restore was part of Planarian’s fork primitive, Planarian does not modify the shared restored image in place, as this could lead to two sandboxes writing the same image concurrently. Instead, Planarian writes a per-sandbox copy of the restored image.

## 6 Remote state management and consistency

Beyond the local sandbox state, Planarian also captures the remote state (§6.1), managing these heterogeneous states consistently via statepoints using the state management primitives (§6.2). Planarian also provides asynchronous management of snapshot and fork, hiding the state management overhead under the LLM latency (§6.3).

## 6.1 Remote state proxy

To make remote state restorable, Planarian routes external tool calls through its state proxy component. This component provides similar functionality to the sandbox manager in that it is responsible for capturing and restoring the remote part of a statepoint. When the agent interacts with an external service that supports versioning, e.g., a branchable database [11, 14], the state proxy can simply use the service’s existing functionality to create agent statepoints and roll back or fork from them. However, Planarian also targets the more general case, where the external service does not support versioning, and some of its operations are not compensable in their original form. To handle this, the state proxy first transforms each supported external tool call into a recoverable form and generates a compensating action, which can be used to undo the original action.

Algorithm 1 Continuous incremental checkpoint/restore   
Input: sandbox ��, checkpoint chain $C = \langle c _ { 0 } , \ldots , c _ { n } \rangle _ { }$ , full-dump   
interval $k ,$ restore target $c _ { r } \in C ,$ primitive ���.   
1: Checkpoint(��,�, �)   
2: $\mathbf { \overline { { i f } } } \ C = \varnothing \ \vee \ | C | \geq k$ then   
3: �<sub>�+1</sub> ← Dump(��, track-mem); $C \gets \langle c _ { n + 1 } \rangle$   
4: else   
5: $c _ { n + 1 }$ ← Dump(��, track-mem, paren $\scriptstyle \mathrm { t = } c _ { n } ) ; C \gets C \cdot c _ { n + 1 }$   
6: end if   
7: Restore(��, �, �<sub>�</sub>, ���)   
8: Assembl $\mathfrak { s } ( \langle c _ { 0 } , \ldots , c _ { r } \rangle )$ from �; if ��� = rollback: Stop(��)   
9: �� ← Restore $( s b , \ c _ { r } , $ hook=pre-resume)   
10: begin hook pre-resume(��)   
11: for all ��� ∈ Processes(��) do   
12: Write(/proc/���/clear\_refs, 4)   
13: end for   
14: � ← 1 + max{ �.start : � ∈ Threads(��) }   
15: wait until Uptime $( ) \geq E ;$ return �   
16: end hook   
17: if ��� = rollback then   
18: �<sub>�</sub>.dump\_time $ E ; C  \langle c _ { 0 } , . . . , c _ { r } \rangle$   
19: else $\mathbf { i f } \ s m p = f o r k$ then   
20: $c _ { r } ^ { \prime } \gets \mathrm { C o p v } ( c _ { r } )$   
21: $c _ { r } ^ { \prime } .$ .dump\_time $ E ; C  \langle c _ { 0 } , . . . , c _ { r - 1 } , c _ { r } ^ { \prime } \rangle$   
22: end if

Proxy integration. To integrate the state proxy with an agent, the agent harness configures the state proxy as the MCP server in place ofthe target service’s MCP server. When the agent issues an MCP call, the request is dispatched from the harness’s MCP client to the state proxy. This proxy is both an MCP server (to the agent harness) and an MCP client (to the target MCP server), transparently intercepting all requests without changing the original MCP APIs.

Request transformation. Upon receiving a request, the proxy first identifies whether the request can be compensated after transformation; unsupported requests that cannot be compensated are rejected before execution. The proxy then transforms the request into a compensable form. The rewrite and compensation logic is specific to the target MCP server. In Planarian, we prototype the state proxy to support compensable SQL operations for the database MCP server. The protocol transforms a single SQL statement into a compensable form: a pre-image statement captures the necessary table state, the original statement mutates the database state, and a post-image statement records the keys afected by the original statement. These three statements are executed as a single SQL transaction, ensuring atomic execution. This makes the original statement compensable: deleted or modified rows can be restored by applying the values in the pre-image table, and added rows can be deleted according to the keys recorded in the post-image table.

```shell
1 // --- statepoint metadata --- //
2 " statepoint -id":
3 " parent - statepoint -id":
4 " branch -id":
5 " timestamp ":
6 " name ":
7 "state - evidence ": {
8 "state - description ":
9 " prior - outcomes ":
10 },
11 // --- statepoint artifacts //
12 "local - state ": {
13 "criu - checkpoint - path ":
14 "zfs - snapshot ":
15 },
16 " remote - state ": {
17 " external - endpoint ":
18 "mcp -call - position ":
19 "hook - path ":
20 },
21 " status ": " discarded | pending | committed "
```  
Fi<sub>g</sub>ure 5. Exam<sub>p</sub>le entr<sub>y</sub> of an a<sub>g</sub>ent state<sub>p</sub>oint.

Compensation generation and enforcement. After rewriting, the state proxy generates the compensating action, forwards the rewritten request to the remote target MCP server, and records the compensating action in the undo log after successful execution. All proxy operations are executed under a proxy-level lock, ensuring that requests are serialized and recorded in a request log, so that subsequent compensating actions can be applied in the exact reverse order of their MCP requests in the log.

## 6.2 Statepoint manager and registry

The statepoint registry manages a collection of the statepoints defined in §3.1. Each registry entry (Fig. 5) includes all three artifacts in a single statepoint: CRIU’s incremental process checkpoint, the zero-copy ZFS snapshot, and the remote state tracked by the state proxy.

To ensure each statepoint is a consistent slice ofthe agent’s environment state that covers all heterogeneous resources that the agent mutates, Planarian proposes a consistent snapshot and rollback/fork protocol on live, running sandboxes and services. At the start of Planarian’s snapshot, the statepoint manager first freezes the container’s processes via cgroup freeze. It then takes the CRIU checkpoint and ZFS snapshot, and captures the remote state through the state proxy. The container’s processes are unfrozen once all captures finish, ensuring that the sandbox state does not drift and thus stays consistent with the other artifacts. The statepoint manager only marks a statepoint as committed once all captures succeed; otherwise the statepoint manager marks it as pending and does not treat it as a valid statepoint. Rollback restores them as a single operation.

The above mechanism lets Planarian maintain a stopthe-world invariant on the container across the entire snapshot window, including CRIU’s internal process dump phase, where the processes remain frozen after the dump and resume only when Planarian explicitly unfreezes the cgroup.

This lets Planarian freeze the container activity during the snapshot without stopping the Podman container, avoiding the cold-start overhead on every snapshot operation.

As Planarian cannot quiesce a running external service that does not expose such APIs, Planarian’s statepoint manager takes snapshots between tool calls. As each MCP server exposes a single service, one tool call typically mutates only one external service. Therefore, under the assumption that the external service is logically isolated by tenant (i.e., the agent-readable and agent-mutable state of the external service is not afected by other tenants), Planarian guarantees consistency across the sandbox and the external services.

On rollback, the statepoint manager first stops the container, then applies the compensating actions via the state proxy and restores the file system. Once all other resources have been rolled back, CRIU starts its restore, making sure the container does not drift during the restore. Fork follows a similar pattern, but if the remote state is only compensable rather than forkable, Planarian prevents the forked sandbox from using the state proxy, which would otherwise cause a consistency issue between exploration branches.

## 6.3 Orchestrating snapshot and execution

Since the agent is waiting for the LLM’s response between tool calls, the state management operations can be run in parallel with the LLM inference request. Planarian is designed as a lightweight, daemonless runtime on top of Podman, CRIU, and ZFS, without requiring a dedicated runtime service. To make both snapshot and fork asynchronous, Planarian launches background workers and coordinates them using a per-sandbox file lock. Tool calls, including command execution and MCP requests, acquire a shared lock, enabling concurrent tool execution. Snapshot and rollback either need the state to be frozen or mutate it, so they acquire an exclusive lock, waiting for in-flight tool calls to finish and preventing new tool calls from executing until they complete. Since fork does not modify the source sandbox, it acquires a shared lock on the source sandbox and an exclusive lock on the forked sandbox, whose state it mutates. Snapshot and fork thus overlap with LLM inference while preserving statepoint consistency.

## 7 Evaluation

We evaluate Planarian to answer the following questions: (i) How does Planarian perform under agentic exploration tasks (§7.2)? (ii) What is Planarian’s performance compared to existing state management approaches on system administration and coding tasks (§7.3)? (iii) What is Planarian’s overhead compared to no state management (§7.4)? (iv) What is the latency of Planarian’s state management primitives (§7.5)?

![](images/6bf50a2241832f8157017dafe01fbeab875386aab90214a4faa89c270d4279ce.jpg)  
Fi<sub>g</sub>ure 6<sub>.</sub> GameWorld<sub>,</sub> score

![](images/ac83fb9dc48392b86e11ab2b9b2e720e1ef92ffad00e2039bb0a9df5ec6bed7c.jpg)  
Fi<sub>g</sub>ure 7. Mario<sub>,</sub> max score b<sub>y</sub> ste<sub>p</sub>

![](images/c9c3848ecaa658ca9650923ac8fc2c1c189e8741b97b8db258003b414287b63d.jpg)  
Fi<sub>g</sub>ure 8<sub>.</sub> MCTS-st<sub>y</sub>le ex<sub>p</sub>loration

## 7.1 Experimental setup

Testbed. We evaluate Planarian on an Ubuntu 22.04 server with Intel Xeon Silver 4310 CPUs (24 cores, 48 threads), 128 GB of memory, and NVMe SSD storage.

Implementation. We implement Planarian in Go and Python. We use Podman 5.8.6 as the agent sandbox, ZFS 2.3.9 for sandbox storage, and CRIU 4.2.1 for process checkpointing. We also use the MCP toolbox for databases [6] as an MCP server. Podman’s container network is enabled when needed.

We integrate Planarian with four agents: (i) Coding—we use mini-swe-agent [42], a lightweight software engineering agent designed for agentic coding tasks; (ii) Personal assistant—we employ nanobot [13], a Python implementation of openclaw [15] that automates everyday tasks; (iii) Exploration harness—we use SWE-search [19], an MCTS-style tree-search harness for coding tasks; and (iv) Gaming—we use the GameWorld [34] harness and benchmark.

Baselines. We compare Planarian against two baselines: (i) No-snapshot, which executes agent tasks in a Podman container without state management. (ii) Full-snapshot, which runs Planarian with Podman’s native snapshot mechanism that exports a full snapshot of the container’s process tree together with its file-system changes. Both baselines use an ext4 filesystem without CoW storage.

Agent workloads. We evaluate Planarian with four realworld agentic workloads: (i) DevOps-Gym [39] consists of 704 DevOps tasks collected from over 30 projects; (ii) Terminal-Bench 2.0 [33] includes 89 tasks, spanning a wide range of categories from terminal environments; (iii) Bird-interactlite [29] is an interactive SQL-generation benchmark with 300 database tasks; and (iv) GameWorld [34] is a gaming benchmark with 34 popular games.

## 7.2 Agentic exploration with state management

LLM-driven exploration. First, we measure how the LLM can proactively interact with Planarian to explore the execution space and safely try diferent rollouts to find better execution plans in fewer steps. For this, we integrate Planarian with GameWorld, enabling the gaming agent to roll back to a previous statepoint. When the game reaches a terminal state in a round, the LLM reviews all available statepoints in the state ledger and either rolls back to a previous statepoint or resumes from the reset game state; after a rollback, the restore context is appended to the following prompts. The state ledger and restore context are constructed from the live game state of the GameWorld harness, with player-unobservable state filtered out.<sup>2</sup> Statepoints are captured between game plays, and only non-terminal states are available for rollback.

We evaluate Planarian on the GameWorld [34] benchmark with three diverse games: Mario, Flappy Bird, and Minecraft. For each, we use the dificulty level at which the agent makes the most absolute progress toward the goal within the step limit. Note that the GameWorld-provided Minecraft game is a simplified item-collection simulator: to approximate the real game’s dificulty, we add real hazards to this simulator, including creepers, lava damage, etc.

As shown in Fig. 6, Planarian enables the LLM to find a better strategy within 100 steps compared to no state management: the agent achieves 15×, 1.4×, and 1.4× the Nosnapshot scores in Mario, Flappy Bird, and Minecraft, respectively. When the game enters a bad state (e.g., a reset due to death), the LLM uses Planarian to roll back to a previous good state, explores alternative plans (as guided by the restore context), and achieves a higher score (see Fig. 7).

Note that snapshots may afect the timing of frames and may change the outcomes. To rule out this efect, we add a Planarian (No rollback) baseline, which uses Planarian to snapshot after each game step but without exposing the rollback primitive. This variant achieves a higher score than No-snapshot, but still lower than full Planarian.

Harness-driven exploration. We integrate Planarian with SWE-search [19], an MCTS-style tree search harness that explores the execution space of software-engineering tasks using Git-based versioning and branching. Since Planarian supports snapshot/fork over the full sandbox state, we extend this harness to MCTS-style tree search on the Terminal-Bench tasks. Nodes in the search tree are snapshotted and forked on demand, overlapping these operations with LLM invocations. We use GPT-5.6-luna (medium reasoning) to generate MCTS-style tree-search traces, and replay the full traces, including the recorded LLM latency, command execution, and snapshot/fork operations on both Planarian and Full-snapshot. We compare Planarian with Fullsnapshot: No-snapshot is not included, because it lacks the fork primitive needed for MCTS-style search.

No-snapshot Full-snapshot Planarian  
![](images/dcd9cb2e055916a153b797aead715befa6945f3825af7b3a556fa1a7755350e3.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 9<sub>.</sub> T<sub>e</sub>rmin<sub>a</sub>l-B<sub>e</sub>n<sub>c</sub>h

![](images/891a753de2af35943bf5eef0c4108c4f857a4c556386bdc4372007ef0acd87b4.jpg)  
Fi<sub>g</sub>ure 10. DevO<sub>p</sub>s-G<sub>y</sub>m

![](images/8bd5c0312c5d97bb397bd190123c3ed1d45ab8f785041f9c276281e2d33cd2ff.jpg)  
Fi<sub>gure</sub> 11<sub>.</sub> Bi<sub>r</sub>d<sub>-</sub>i<sub>n</sub>t<sub>erac</sub>t<sub>-</sub>lit<sub>e</sub>

![](images/a5b30d44d90f9e664fdbe3db4e9432cbd5cdd145ae87dea32afaa7becc5ff6a8.jpg)

![](images/abf4a5ec8a15a941d5e9c55295860b9dd4345d9775d8b3cdb03a509ea8a2724d.jpg)  
(a) Snaps<sup>h</sup>ot be<sup>f</sup>ore ro<sup>ll</sup>bac<sup>k</sup>/<sup>f</sup>or<sup>k</sup>

![](images/804062796454f7f4035b2e6de800501f2188d624345d84cfbd54d29755f38dbc.jpg)  
(b) Snaps<sup>h</sup>ot after ro<sup>ll</sup>bac<sup>k</sup>

![](images/11c05104bf3e9520ba43500fb5dc89cf1aab3c1297dad9f6258fc828fd0309e7.jpg)  
(c) Snaps<sup>h</sup>ot after <sup>f</sup>or<sup>k</sup>

![](images/843a14d772e999198521769750e980fae1a68bb3d5a65d456eb008beb04bd87d.jpg)

Fi<sub>g</sub>ure 12. Sna<sub>p</sub>shot cost  
![](images/e791ff727042fec322ed77e62a46e1ca8e93ff1b6a2c4e1d702adf0f4a10b789.jpg)  
Fi<sub>gure</sub> 13<sub>.</sub> R<sub>o</sub>llb<sub>ac</sub>k <sub>cos</sub>t

![](images/ec45868ebd654f4a9d6ffad327816e666029a43cac1227c68eb8de31096f5ba5.jpg)

![](images/032562a8cf065a390b22a1da1f9aec13c6c544700b6daddfef7c45ed7c28db34.jpg)  
Fi<sub>g</sub>ure 14. Fork cost

Fig. 8 shows the overhead of MCTS-style tree search with Planarian and Full-snapshot on nine system-administration tasks from Terminal-Bench, broken down into fork, snapshot, command execution, and end-to-end (E2E) time; the latter partially overlaps with snapshot and fork. Full-snapshot’s primitives are significantly more expensive: its snapshot overhead is 10× that of Planarian, because Planarian checkpoints incrementally after a process restore, and its fork overhead is 2× that of Planarian. As a result, Fullsnapshot is 9% slower than Planarian in end-to-end time.

The end-to-end gap is smaller than the primitive gap, because Full-snapshot also benefits from overlap: each node in SWE-search issues two LLM calls, and the resulting LLM latency is long enough to hide most of Full-snapshot’s overhead. With more frequent snapshot/fork operations or with lower LLM latency, less of that overhead can be hidden, and the end-to-end gap grows toward the performance gap of the snapshot/fork primitives.

## 7.3 Safe execution with state management

We next evaluate Planarian’s performance when used for user-driven rollback. We first use the Planarian-integrated agent with GPT-5.6-luna (medium reasoning) to generate tool-calling traces, then replay these traces on Planarian and the two baselines, because LLM outputs are non-determinis Terminal. In this experiment, the mini-swe-agent runs on Planarian and executes tasks from Terminal-Bench (nine tasks from the system-administration category). Planarian snapshots the agent sandbox after each tool call, aligning with the granularity of human approval. At the end of execution, both Planarian and Full-snapshot roll back to a previous snapshot.

Fig. 9 shows that, compared with No-snapshot, Planarian adds a 1% overhead, which is significantly faster than Fullsnapshot (20%), because Full-snapshot’s snapshots are more heavyweight. Compared with Full-snapshot, Planarian reduces raw snapshot/rollback times by 89%. Most Terminal-Bench tasks are not filesystem heavy, so copy-onwrite does not add significant overhead.

DevOps. We integrate mini-swe-agent with Planarian to evaluate performance on DevOps tasks. We use DevOps Gym’s 16 implementation tasks in the build/configuration category. Fig. 10 shows that Planarian requires less than 3% extra task completion time compared to No-snapshot; Full-snapshot adds a 59% overhead. Since Full-snapshot’s sandbox snapshot/rollback time is ∼24× that of Planarian, it becomes challenging to hide the snapshot time behind the LLM inference time.

## 7.4 Overhead of consistent state management

To evaluate Planarian’s overhead when both local and remote state are managed consistently, we execute 50 database management tasks from the Bird-interact-lite [29] benchmark, in which an agent mutates a PostgreSQL database via MCP [6]. For Planarian, the state proxy generates compensating actions and rewrites SQL requests. We use GPT-5.6-luna (medium reasoning) to generate traces for MCP requests. As in §7.3, we then replay the requests on both No-snapshot and Planarian, with Planarian snapshotting after each request and rolling back the sandbox and database.

Fig. 11 shows that, while managing local and remote state consistently, Planarian adds a 3% overhead to the end-toend time. This overhead comes from three sources: (i) when interacting with the MCP database server, the harness sometimes sends several requests instead of interleaving them with LLM inference, preventing snapshot time from being hidden; (ii) Planarian incurs extra rollback operations for both the sandbox and database state; and (iii) to make MCP requests revertible, the state proxy adds SQL operations: Planarian takes 1.3× the MCP round-trip time compared to the No-snapshot baseline. Since agentic database operations and MCP requests have low latency, this request amplification only increases end-to-end time by less than 0.3%.

## 7.5 State management primitives

We also evaluate the overhead of Planarian’s state management primitives under diferent microbenchmarks. We use a workload that dirties memory or the file system. We take two successive snapshots and measure the overhead of the second, which is incremental when supported. We then measure the overhead of rolling back to, or forking from, this second statepoint. For the memory workload, we also measure the latency of the first snapshot taken after the rollback and after the fork. We compare Planarian with Full-snapshot and with CRIU-inc, which wires CRIU’s native incremental checkpointing through Podman.

Fig. 12 shows the overhead under these workloads. With a large working set (8 GB), Planarian’s incremental snapshot achieves 74× and 231× speedups over Full-snapshot on the memory and file system workloads, respectively, because Full-snapshot always provides a full dump. Planarian’s continuous incremental checkpoint protocol is also faster than CRIU’s native incremental checkpointing for the first snapshot after rollback: on the 8 GB memory workload, Planarian provides an 11× speedup over CRIU-inc, because Planarian continues to checkpoint incrementally after a restore.

We also measure the overhead of the rollback and fork primitives on the incremental snapshot (see Figs. 13 and 14). For rollback, Planarian achieves 4× and 67× speedups over Full-snapshot on the 8 GB memory and file system workloads, respectively. Restoring the filesystem is faster than restoring memory, due to the cost of reconstructing a large process. Fork shows a similar pattern: for a sandbox with the same memory and file system state, Planarian exhibits 4× and 61× speedups over the same baseline. We conclude that, with the incremental checkpoint protocol and CoW storage, Planarian provides eficient state management primitives.

## 8 Related Work

State recovery for agents. Existing systems version the state an agent mutates, either to recover the system from the agent’s mistakes or to make the agent’s state portable across machines. AgentFS [40] is a SQLite-backed overlay file system that makes the agent’s file system state checkpointable and portable. It covers file system state only, and a checkpoint copies the entire writable overlay layer. Daytona [9] and E2B [12] are cloud sandbox providers that ofer persistent sandbox snapshots from which new sandboxes are created. They focus on the sandbox state itself, leaving remote state out of scope. Similarly, Agent Sandbox [32] supports persistent, stateful execution of Kubernetes (k8s) pods, so that idle agent sandboxes can be paused. k8s is designed to host remotely deployed microservices, whereas Planarian targets the user’s local deployment; Planarian can integrate with Agent Sandbox to make such remotely deployed sandboxes recoverable.

Other work makes external database state recoverable. Dolt [11] provides Git-like operations on database state, enabling users to revert, branch, and merge the agent-mutated database. Similarly, Neon [14] provides a CoW branching primitive for PostgreSQL. Both branchable database services can be integrated with Planarian through its hooks, enabling consistent rollback and branching across both local and remote state.

Agentic explorations. SWE-search [19] is an MCTS-style search that executes, evaluates, and expands high-reward nodes, enabling search over Git-based software engineering tasks. Planarian extends this beyond Git, enabling tree search on arbitrary terminal tasks. ExACT [44] provides LLM-driven MCTS exploration for GUI tasks and relies on backtracking in the web environment. Planarian can integrate with ExACT, enabling backtracking even when the workload does not expose this capability.

Snapshot/restore systems. Various snapshot and restore systems support capturing a live system state and rollback when needed. Checkpoint/Restore In Userspace (CRIU) [7] is a userspace process checkpointing tool that captures the state of a process tree, including memory pages, active sockets, and file descriptors, into a restorable image. As an alternative, DMTCP [18] is a process checkpointing library that dumps process state by preloading its library into the application at launch. We use CRIU for process checkpointing in Planarian since it is widely integrated with OCI runtimes. Snapshots are also natively supported by many file systems and device mappers. The Zettabyte File System (ZFS) [17] is a CoW file system that pools the available storage and manages the pool state as a Merkle tree. Since it provides block-level CoW, Planarian integrates ZFS for fast file system snapshots. Other CoW snapshot mechanisms for persistent state include Btrfs [35], a Linux-native file system that provides CoW over a B-tree, and dm-snapshot [10], a kernel device mapper that provides fast snapshots of a block device.

## 9 Conclusions

We presented Planarian, an agent runtime that manages local and remote environment state through agent statepoints: consistent, restorable point-in-time versions of the environment. Three primitives—snapshot, rollback, and fork—give users, harnesses, and agents a unified interface for recovery and exploration, while the context library keeps agents informed of restored state and past outcomes. Our evaluation shows that Planarian improves task quality by enabling agents to explore alternatives and lets users recover from erroneous agent actions with little overhead.

## References

[1] 2017. CRIU Issue #401. htps://github.com/checkpoint-restore/criu issues/401. Accessed: 2026-08-26.

[2] 2024. Model Context Protocol. htps://modelcontextprotocol.io. Ac cessed: 2026-6-2.

[3] 2025. Claude Code. htps://www.anthropic.com/product/claude-code. Accessed: 2026-6-2.

[4] 2025. Codex. htps://openai.com/codex/. Accessed: 2026-6-2.

[5] 2025. Langchain. htps://github.com/langchain-ai/langchain. Ac cessed: 2026-6-2.

[6] 2025. MCP Toolbox for Databases. htps://github.com/googleapis/mcptoolbox. Accessed: 2026-6-2.

[7] 2026. CRIU: Checkpoint/Restore In Userspace. htps://criu.org. Accessed: 2026-08-26.

[8] 2026. Cube Sandbox. htps://github.com/tencentcloud/CubeSandbox. Accessed: 2026-6-2.

[9] 2026. Daytona documentation: Sandboxes and Snapshots. htps:// www.daytona.io/docs/en/. Accessed: 2026-09-19.

[10] 2026. Device-mapper snapshot support. htps://docs.kernel.org/adminguide/device-mapper/snapshot.html. Accessed: 2026-09-19.

[11] 2026. Dolthub documentation: what is Dolt? htps://www.dolthub. com/docs/introduction/what-is-dolt/. Accessed: 2026-09-19.

[12] 2026. E2B. htps://e2b.dev/. Accessed: 2026-6-25.

[13] 2026. Nanobot. htps://github.com/HKUDS/nanobot. Accessed: 2026-6-2.

[14] 2026. Neon documentation. htps://neon.com/docs/introduction. Accessed: 2026-09-19.

[15] 2026. OpenClaw. htps://github.com/openclaw/openclaw. Accessed: 2026-6-2.

[16] 2026. Podman. htps://podman.io/. Accessed: 2026-6-2.

[17] Matt Ahrens, Jef Bonwick, Val Henson, Mark Maybee, and Mark Shellenbaum. 2003. The zettabyte file system. In Proc. ofthe 2nd Usenix Conference on File and Storage Technologies.

[18] Jason Ansel, Kapil Arya, and Gene Cooperman. 2009. DMTCP: Transparent Checkpointing for Cluster Computations and the Desktop. In Proc. ofthe IEEE International Symposium on Parallel & Distributed Processing (IPDPS).

[19] Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Wang. 2025. Swe-search: Enhancing software agents with monte carlo tree search and iterative refinement. In International Conference on Learning Representations.

[20] Mahesh Balakrishnan, Ashwin Bharambe, Davide Testuggine, David Geraghty, David Mao, Vidhya Venkat, Ilya Mironov, Rithesh Baradi, Gayathri Aiyer, and Victoria Dudin. 2026. LogAct: Enabling Agentic Reliability via Shared Logs. arXiv preprint arXiv:2604.07988 (2026).

[21] Edward Y. Chang and Longling Geng. 2025. SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning. Proc. VLDB Endow. 18, 12 (Aug. 2025), 4874–4886. doi:10.14778/3750601.3750611

[22] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).

[23] Zheng Chen, Hanqing Liu, Duling Xu, Dong Dong, Jialin Li, Bangzheng Pu, and Jidong Zhai. 2026. Cordon: Semantic Transactions for Tool-Using LLM Agents. arXiv preprint arXiv:2606.17573 (2026).

[24] Yunpeng Dong, Jingkai He, Yuze Hou, Dong Du, Zhonghu Xu, Si Yu, Yubin Xia, and Haibo Chen. 2026. DeltaBox: Scaling Stateful AI Agents with Millisecond-Level Sandbox Checkpoint/Rollback. arXiv preprint arXiv:2605.22781 (2026).

[25] Dong Du, Tianyi Yu, Yubin Xia, Binyu Zang, Guanglu Yan, Chenggang Qin, Qixuan Wu, and Haibo Chen. 2020. Catalyzer: Sub-millisecond startup for serverless computing with initialization-less booting. In Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 467–481.

[26] Aaron Fanous, Jacob Goldberg, Ank Agarwal, Joanna Lin, Anson Zhou, Sonnet Xu, Vasiliki Bikia, Roxana Daneshjou, and Sanmi Koyejo. 2025. Syceval: Evaluating llm sycophancy. In Proceedings ofthe AAAI/ACM Conference on AI, Ethics, and Society, Vol. 8. 893–900.

[27] Xingyuan Hua, Sheng Yue, and Ju Ren. 2026. Learning to Explore: Scaling Agentic Reasoning via Exploration-Aware Policy Optimization. arXiv preprint arXiv:2605.08978 (2026).

[28] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM transactions on information systems 43, 2 (2025), 1–55.

[29] Nan Huo, Xiaohan Xu, Jinyang Li, Per Jacobsson, Shipei Lin, Bowen Qin, Binyuan Hui, Xiaolong Li, Ge Qu, Shuzheng Si, Linheng Han, Edward Alexander, Xintong Zhu, Rui Qin, Ruihan Yu, Yiyao Jin, Feige Zhou, Weihao Zhong, Yun Chen, Hongyu Liu, Chenhao Ma, Fatma Ozcan, Yannis Papakonstantinou, and Reynold Cheng. 2026. BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions. In The Fourteenth International Conference on Learning Representations. htps://openreview.net/forum?id=nHrYBGujps

[30] Taesoo Kim, Xi Wang, Nickolai Zeldovich, and M Frans Kaashoek. 2010. Intrusion recovery using selective re-execution. In 9th USENIX Symposium on Operating Systems Design and Implementation (OSDI 10).

[31] Ákos Kovács. 2017. Comparison of diferent Linux containers. In 2017 40th International Conference on Telecommunications and Signal

Processing (TSP). IEEE, 47–51.

[32] Kubernetes SIGs. 2026. Agent Sandbox: isolated, stateful, singleton workloads on Kubernetes. htps://github.com/kubernetes-sigs/agentsandbox. Accessed: 2026-09-19.

[33] Mike A Merrill, Alexander G Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E Kelly Buchanan, et al. 2026. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868 (2026).

[34] Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, and Mike Zheng Shou. 2026. Gameworld: Towards standardized and verifiable evaluation of multimodal game agents. In European Conference on Computer Vision. Springer, 41–59.

[35] Ohad Rodeh, Josef Bacik, and Chris Mason. 2013. BTRFS: The Linux B-tree filesystem. ACM Transactions on Storage (TOS) 9, 3 (2013), 1–32.

[36] Rui Shu, Peipei Wang, Sigmund A Gorski III, Benjamin Andow, Adwait Nadkarni, Luke Deshotels, Jason Gionta, William Enck, and Xiaohui Gu. 2016. A study of security isolation techniques. ACM Computing Surveys (CSUR) 49, 3 (2016), 1–37.

[37] Ya-Yunn Su, Mona Attariyan, and Jason Flinn. 2007. AutoBash: improving configuration management with operating system causality analysis. In Proceedings ofTwenty-First ACM SIGOPS Symposium on Operating Systems Principles (Stevenson, Washington, USA) (SOSP ’07). Association for Computing Machinery, New York, NY, USA, 237–250. doi:10.1145/1294261.1294284

[38] Maciej Świechowski, Konrad Godlewski, Bartosz Sawicki, and Jacek Mańdziuk. 2023. Monte Carlo tree search: A review of recent modi fications and applications. Artificial Intelligence Review 56, 3 (2023), 2497–2562.

[39] Yuheng Tang, Kaijie Zhu, Bonan Ruan, Chuqi Zhang, Michael Yang, Hongwei Li, Suyue Guo, Tianneng Shi, Zekun Li, Christopher Kruegel, et al. 2026. DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle. arXiv preprint arXiv:2601.20882 (2026).

[40] Turso. 2026. AgentFS: The filesystem for agents. htps://github.com/ tursodatabase/agentfs. Accessed: 2026-09-19.

[41] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. 2023. Autogen: Enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155 (2023).

[42] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. htps://arxiv.org/abs/2405.15793

[43] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR).

[44] Xiao Yu, Baolin Peng, Vineeth Vajipey, Hao Cheng, Michel Galley, Jianfeng Gao, and Zhou Yu. 2025. Exact: Teaching ai agents to explore with reflective-mcts and exploratory learning. In International Conference on Learning Representations.

[45] Haoran Zhang, Adney Cardoza, Peter Baile Chen, Sebastian Angel, and Vincent Liu. 2020. Fault-tolerant and transactional stateful serverless workflows. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). 1187–1204.

[46] Yusheng Zheng, Yiwei Yang, Wei Zhang, and Andi Quinn. 2026. ACR-Fence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore. arXiv preprint arXiv:2603.20625 (2026).