import prompt_graph.data
import prompt_graph.model
import prompt_graph.pretrain
import prompt_graph.prompt
import prompt_graph.tasker
import prompt_graph.utils

# RecBole-style registry: register built-in prompts/evaluators so new prompts can plug in without modifying core code
from prompt_graph.registry_config import setup_registry
setup_registry()
