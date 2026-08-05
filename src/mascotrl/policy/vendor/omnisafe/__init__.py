# Vendored from OmniSafe (Apache-2.0).
# Upstream commit: 15603dd7a654a991d0a4648216b69d60b81a6366
"""OmniSafe dual modules without Safety-Gymnasium."""
from mascotrl.policy.vendor.omnisafe.lagrange import Lagrange
from mascotrl.policy.vendor.omnisafe.pid_lagrange import PIDLagrangian

__all__ = ["Lagrange", "PIDLagrangian"]
