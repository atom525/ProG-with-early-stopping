"""
Registration of all built-in prompts, evaluators, and data modes.
Add new prompts here (or in a separate file that imports this) - no need to touch task code.
"""
from .registry import PromptRegistry, DataMode
from .evaluation import GNNNodeEva, GNNGraphEva, GPPTEva, GPPTGraphEva, GPFEva, GpromptEva, AllInOneEva, MultiGpromptEva


def _register_evaluators():
    """Register evaluator adapters for each (prompt_type, task_type)."""

    # ---- NodeTask evaluators ----
    def node_none_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GNNNodeEva(data, idx, gnn, answering, output_dim, device)

    def node_gppt_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GPPTEva(data, idx, gnn, prompt, output_dim, device)

    def node_graph_loader_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, center=None, **extra):
        # GPF, GPF-plus, All-in-one: use loader
        if 'allinone' in str(extra.get('_hint', '')).lower():
            return AllInOneEva(loader, prompt, gnn, answering, output_dim, device)
        if center is not None:
            return GpromptEva(loader, gnn, prompt, center, output_dim, device)
        return GPFEva(loader, gnn, prompt, answering, output_dim, device)

    def node_allinone_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return AllInOneEva(loader, prompt, gnn, answering, output_dim, device)

    def node_gpf_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GPFEva(loader, gnn, prompt, answering, output_dim, device)

    def node_gprompt_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, center=None, **extra):
        return GpromptEva(loader, gnn, prompt, center, output_dim, device)

    def node_multigprompt_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        test_embs = extra.get('test_embs')
        test_lbls = extra.get('test_lbls')
        idx_test = extra.get('idx_test')
        prompt_feature = extra.get('prompt_feature')
        Preprompt = extra.get('Preprompt')
        DownPrompt = extra.get('DownPrompt')
        sp_adj = extra.get('sp_adj')
        return MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, Preprompt, DownPrompt, sp_adj, output_dim, device)

    PromptRegistry.register_evaluator('None', 'NodeTask', node_none_eval)
    PromptRegistry.register_evaluator('GPPT', 'NodeTask', node_gppt_eval)
    PromptRegistry.register_evaluator('GPF', 'NodeTask', node_gpf_eval)
    PromptRegistry.register_evaluator('GPF-plus', 'NodeTask', node_gpf_eval)
    PromptRegistry.register_evaluator('All-in-one', 'NodeTask', node_allinone_eval)
    PromptRegistry.register_evaluator('Gprompt', 'NodeTask', node_gprompt_eval)
    PromptRegistry.register_evaluator('MultiGprompt', 'NodeTask', node_multigprompt_eval)

    # ---- GraphTask evaluators ----
    def graph_none_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GNNGraphEva(loader, gnn, answering, output_dim, device)

    def graph_gppt_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GPPTGraphEva(loader, gnn, prompt, output_dim, device)

    def graph_gpf_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return GPFEva(loader, gnn, prompt, answering, output_dim, device)

    def graph_gprompt_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, center=None, **extra):
        return GpromptEva(loader, gnn, prompt, center, output_dim, device)

    def graph_allinone_eval(loader, data, idx, gnn, prompt, answering, output_dim, device, **extra):
        return AllInOneEva(loader, prompt, gnn, answering, output_dim, device)

    PromptRegistry.register_evaluator('None', 'GraphTask', graph_none_eval)
    PromptRegistry.register_evaluator('GPPT', 'GraphTask', graph_gppt_eval)
    PromptRegistry.register_evaluator('GPF', 'GraphTask', graph_gpf_eval)
    PromptRegistry.register_evaluator('GPF-plus', 'GraphTask', graph_gpf_eval)
    PromptRegistry.register_evaluator('Gprompt', 'GraphTask', graph_gprompt_eval)
    PromptRegistry.register_evaluator('All-in-one', 'GraphTask', graph_allinone_eval)


def _register_prompts():
    """Register prompt classes and metadata."""
    from .prompt import GPF, GPF_plus, GPPTPrompt, HeavyPrompt, Gprompt

    # None
    PromptRegistry.register_prompt('None', None, DataMode.NODE_FULL, needs_center=False, train_method='train' if False else None)

    # GPPT
    def gppt_init(task):
        center_num = task.output_dim
        if task.task_type == 'NodeTask' and task.dataset_name == 'Texas':
            center_num = 5
        return {'n_hidden': task.hid_dim, 'center_num': center_num, 'n_classes': task.output_dim, 'device': task.device}

    PromptRegistry.register_prompt('GPPT', GPPTPrompt, DataMode.NODE_FULL, needs_center=False, train_method='GPPTtrain',
                                  init_kwargs_fn=gppt_init)

    # All-in-one
    PromptRegistry.register_prompt('All-in-one', HeavyPrompt, DataMode.GRAPH_INDUCED, needs_center=False,
                                  train_method='AllInOneTrain',
                                  init_kwargs_fn=lambda t: {'token_dim': t.input_dim, 'token_num': 10, 'cross_prune': 0.1, 'inner_prune': 0.3})

    # GPF (GPF uses in_channels, not input_dim)
    PromptRegistry.register_prompt('GPF', GPF, DataMode.GRAPH_INDUCED, needs_center=False, train_method='GPFTrain',
                                   init_kwargs_fn=lambda t: {'in_channels': t.input_dim})

    # GPF-plus (GPF_plus uses in_channels and p_num, not input_dim and token_num)
    PromptRegistry.register_prompt('GPF-plus', GPF_plus, DataMode.GRAPH_INDUCED, needs_center=False, train_method='GPFTrain',
                                   init_kwargs_fn=lambda t: {'in_channels': t.input_dim, 'p_num': 20})

    # Gprompt (input_dim in constructor is actually GNN hid_dim)
    PromptRegistry.register_prompt('Gprompt', Gprompt, DataMode.GRAPH_INDUCED, needs_center=True, train_method='GpromptTrain',
                                   init_kwargs_fn=lambda t: {'input_dim': t.hid_dim})

    # MultiGprompt - special, has custom init in BaseTask (uses Preprompt, feature_prompt, DownPrompt)
    PromptRegistry.register_prompt('MultiGprompt', None, DataMode.MULTI_SPECIAL, needs_center=False, train_method='MultiGpromptTrain')
    # Note: MultiGprompt init is done manually in task.py due to Preprompt/featureprompt/downprompt wiring


def setup_registry():
    """Call once at startup to register all built-in components."""
    _register_evaluators()
    _register_prompts()
