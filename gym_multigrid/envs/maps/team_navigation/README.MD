# Field map usage

- Use VS Code's "Edit CSV" extension to edit these maps easily.

- You can choose the team's inital state in the first room or the fruit spawn locations. If not specified in the map, the env will randomly spawn them.

- agent
    - (a, agent_level, agent_idx)

- goal
    -  g = any agent can reach this goal, the room's task ends when all agents reach a goal
    - (g, agent_idx) = a goal state assigned to agent with index agent_idx

- door
    - d, wall that disappears when agents reach the goal states after completing all other tasks in room with index room_idx

- permanent wall
    - w

## High-level navigation tasks

Hierarchical team navigation can load a YAML file through the `navigation_config`
environment argument. The file may define an arbitrary directed task graph under
`mdp.transitions` and one navigation task per directed edge under `tasks`.
Each task uses `from_state`, `to_state`, one goal position per agent, and an
`init_state_dist`. See `gym_multigrid/envs/maps/navigation_branching.yaml` for a
branching example where two tasks merge into a final state.
