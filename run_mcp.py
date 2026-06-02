from mcp.server.fastmcp.server import Settings as _FastMCPSettings

_original_init = _FastMCPSettings.__init__

def _patched_init(self, **kwargs):
    kwargs["host"] = "0.0.0.0"
    _original_init(self, **kwargs)

_FastMCPSettings.__init__ = _patched_init

import runpy
runpy.run_path("/app/mcp_server.py", run_name="__main__")
