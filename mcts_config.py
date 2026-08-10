from dataclasses import dataclass

from app_config import (
    MCTS_ALPHA,
    MCTS_BETA,
    MCTS_D,
    MCTS_DELTA_MAX,
    MCTS_ETA,
    MCTS_GAMMA,
    MCTS_GAMMA_D,
    MCTS_K,
    MCTS_LAMBDA_PUCT,
    MCTS_N_SIM,
    MCTS_ROLLOUT_CALL_CAP_PER_SEARCH,
)


@dataclass
class MCTSConfig:
    N_sim: int = MCTS_N_SIM
    K: int = MCTS_K
    D: int = MCTS_D
    eta: int = MCTS_ETA
    # Optional hard cap for rollout-model calls in one search. Zero disables it.
    max_rollout_calls_per_search: int = MCTS_ROLLOUT_CALL_CAP_PER_SEARCH
    alpha: float = MCTS_ALPHA
    beta: float = MCTS_BETA
    # Eq. (4) uninformative-action penalty.
    gamma: float = MCTS_GAMMA
    # Eq. (6) cumulative-return discount; distinct from the penalty coefficient.
    gamma_d: float = MCTS_GAMMA_D
    lambda_puct: float = MCTS_LAMBDA_PUCT
    delta_max: float = MCTS_DELTA_MAX

    def __post_init__(self):
        self.N_sim = max(1, int(self.N_sim))
        self.K = max(1, int(self.K))
        self.D = max(0, int(self.D))
        self.eta = max(1, int(self.eta))
        self.max_rollout_calls_per_search = max(0, int(self.max_rollout_calls_per_search))
        self.alpha = max(0.0, float(self.alpha))
        self.beta = max(0.0, float(self.beta))
        self.gamma = max(0.0, float(self.gamma))
        self.gamma_d = max(0.0, min(1.0, float(self.gamma_d)))
        self.lambda_puct = max(0.0, float(self.lambda_puct))
