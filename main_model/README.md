# Driving Model

This folder is for PyTorch model code only.

It is intentionally separate from `gta_stream/`, which is the game I/O layer:

- `gta_stream/` handles GTA V state export, control input, and the WebSocket bridge
- `driving_model/` handles neural network code, training code, and model utilities

Current files:

- `main_model.py`: main embedding + attention model, now calling the action planner
- `action_planner.py`: one-function planner that stops at the final action tensor

The main model reads raw saved GTA JSON frames from `../gta_stream/stats/` for
smoke tests, but the model code itself stays isolated in this folder.

Game execution, live training, labels, and checkpoints live outside this folder.
