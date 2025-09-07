from terminal_bench.cli.tb.main import app
from terminal_bench.harness import harness
from minienv_terminal import spin_up_minienv_terminal

def patched_spin_up_terminal(*args, **kwargs):
    return spin_up_minienv_terminal(
        **kwargs, 
        beaker_workspace='ai2/rollouts'
    )

harness.spin_up_terminal = patched_spin_up_terminal

app()
