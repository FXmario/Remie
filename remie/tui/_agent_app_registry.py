"""Deferred registration of AgentApp for circular-import-free use."""

_modules: list[str] = []
_agent_app = None


def register_module(module_name: str) -> None:
    """Record a module that needs AgentApp injected later."""
    _modules.append(module_name)


def register(agent_app) -> None:
    """Inject AgentApp into all registered modules' globals."""
    global _agent_app
    import sys

    _agent_app = agent_app
    for module_name in _modules:
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, 'AgentApp', agent_app)
