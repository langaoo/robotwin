from __future__ import annotations

import DP2DP3.deploy_film_drifting_policy as base
from features_common.depth_guided_film_drifting.policy_drifting_v18_raw_arep import (
	DA3FilmDriftingPolicyV18RawAREP,
)


# Reuse the stable drifting deploy path, but swap the multi-temp class symbol
# so checkpoints that route through the multi-temp branch instantiate RawAREP.
base.DA3FilmDriftingPolicyV18MultiTemp = DA3FilmDriftingPolicyV18RawAREP

encode_obs = base.encode_obs
eval = base.eval
reset_model = base.reset_model
get_model = base.get_model
