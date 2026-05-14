import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    SIMULATIONS_DIR,
    Info,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import Action, EnvAssertion, RewardType, Task
from tau2.environment.toolkit import ToolType
from tau2.orchestrator.modes import CommunicationMode
from tau2.utils.utils import get_now


class DBEvaluation(BaseModel):
    """Reward-free database evaluation result."""

    db_match: bool


class EnvAssertionEvaluation(BaseModel):
    """Reward-free environment assertion evaluation result."""

    env_assertion: EnvAssertion
    met: bool


class ActionEvaluation(BaseModel):
    """Reward-free action evaluation result."""

    action: Action
    action_match: bool
    tool_type: Optional[ToolType] = Field(
        description="The type of tool (read/write/think/generic).",
        default=None,
    )


class CommunicateEvaluation(BaseModel):
    """Reward-free communication evaluation result."""

    info: str
    met: bool
    justification: str


class NLAssertionEvaluation(BaseModel):
    """Reward-free natural-language assertion evaluation result."""

    nl_assertion: str
    met: bool
    justification: str


class CheckResult(BaseModel):
    """Normalized component score used for evaluation-only aggregation."""

    name: str
    score: float
    passed_count: int
    total_count: int
    passed: Optional[bool] = None


class EvaluationReport(BaseModel):
    """Reward-free evaluation details for a single simulation."""

    domain: str
    task_id: str
    simulation_id: str
    termination_reason: TerminationReason
    mode: str = Field(default=CommunicationMode.HALF_DUPLEX.value)
    evaluation_type: str
    reward_basis: Optional[list[RewardType]] = Field(default=None)
    db_check: Optional[DBEvaluation] = Field(default=None)
    env_assertions: Optional[list[EnvAssertionEvaluation]] = Field(default=None)
    action_checks: Optional[list[ActionEvaluation]] = Field(default=None)
    communicate_checks: Optional[list[CommunicateEvaluation]] = Field(default=None)
    nl_assertions: Optional[list[NLAssertionEvaluation]] = Field(default=None)
    info: Optional[dict] = Field(default=None)


class EvaluationOutcome(BaseModel):
    """Scalar and component-level summary derived from an EvaluationReport."""

    score_policy: str
    overall_score: float
    component_scores: list[CheckResult] = Field(default_factory=list)
    info: Optional[dict] = Field(default=None)


class EvaluatedSimulation(BaseModel):
    """Simulation plus reward-free evaluation artifacts."""

    simulation: SimulationRun
    evaluation_report: EvaluationReport
    evaluation_outcome: EvaluationOutcome

    @property
    def id(self) -> str:
        return self.simulation.id

    @property
    def task_id(self) -> str:
        return self.simulation.task_id

    @property
    def trial(self) -> Optional[int]:
        return self.simulation.trial

    @trial.setter
    def trial(self, value: Optional[int]) -> None:
        self.simulation.trial = value

    @property
    def seed(self) -> Optional[int]:
        return self.simulation.seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        self.simulation.seed = value

    @property
    def termination_reason(self) -> TerminationReason:
        return self.simulation.termination_reason

    @property
    def agent_cost(self) -> Optional[float]:
        return self.simulation.agent_cost

    @property
    def duration(self) -> float:
        return self.simulation.duration


class EvaluationIndexEntry(BaseModel):
    """Lightweight summary entry for directory-format evaluated results."""

    id: str
    task_id: int | str
    trial: int
    overall_score: float | None = None
    termination_reason: str | None = None
    agent_cost: float | None = None
    duration: float | None = None


class EvaluatedResults(BaseModel):
    """Batch container for evaluation-only runs."""

    timestamp: Optional[str] = Field(default_factory=get_now)
    info: Info
    tasks: list[Task]
    simulations: list[EvaluatedSimulation] = Field(default_factory=list)
    simulation_index: list[EvaluationIndexEntry] | None = Field(default=None)
    post_evaluation_mode: Literal["evaluation_only"] = Field(
        default="evaluation_only"
    )
    evaluation_type: str = Field(default="all")
    score_policy: str = Field(default="evaluation_mean_v1")

    @staticmethod
    def _detect_format(path: Path) -> Literal["json", "dir"]:
        path = Path(path)
        if path.is_dir():
            return "dir"
        sims_dir = path.parent / SIMULATIONS_DIR
        if sims_dir.is_dir():
            return "dir"
        return "json"

    @staticmethod
    def _resolve_paths(path: Path) -> tuple[Path, Path]:
        path = Path(path)
        if path.is_dir():
            return path / "results.json", path / SIMULATIONS_DIR
        return path, path.parent / SIMULATIONS_DIR

    def _build_simulation_index(self) -> list[EvaluationIndexEntry]:
        return [
            EvaluationIndexEntry(
                id=sim.id,
                task_id=sim.task_id,
                trial=sim.trial,
                overall_score=sim.evaluation_outcome.overall_score,
                termination_reason=sim.termination_reason,
                agent_cost=sim.agent_cost,
                duration=sim.duration,
            )
            for sim in self.simulations
        ]

    @classmethod
    def load(cls, path: Path) -> "EvaluatedResults":
        path = Path(path)
        fmt = cls._detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                return cls.model_validate_json(f.read())

        meta_path, sims_dir = cls._resolve_paths(path)
        with open(meta_path, "r") as f:
            meta = json.loads(f.read())

        meta.pop("format_version", None)

        simulations = []
        if sims_dir.exists():
            for sim_file in sorted(sims_dir.glob("*.json")):
                with open(sim_file, "r") as f:
                    simulations.append(json.loads(f.read()))

        index = meta.get("simulation_index")
        if index is not None:
            indexed_ids = {entry["id"] for entry in index}
            on_disk_ids = (
                {f.stem for f in sims_dir.glob("*.json")}
                if sims_dir.exists()
                else set()
            )
            missing = indexed_ids - on_disk_ids
            extra = on_disk_ids - indexed_ids
            errors = []
            if missing:
                errors.append(f"Missing simulation files: {sorted(missing)}")
            if extra:
                errors.append(f"Extra simulation files not in index: {sorted(extra)}")
            if errors:
                raise ValueError(
                    f"Dir format integrity check failed for {meta_path}: "
                    + "; ".join(errors)
                )

        meta["simulations"] = simulations
        return cls.model_validate(meta)

    def save(self, path: Path, format: Literal["json", "dir"] = "json") -> None:
        path = Path(path)
        if format == "json":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(self.model_dump_json(indent=2))
            return

        meta_path, sims_dir = self._resolve_paths(path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        sims_dir.mkdir(parents=True, exist_ok=True)

        self.simulation_index = self._build_simulation_index()
        meta = self.model_dump(mode="json", exclude={"simulations"})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        for sim in self.simulations:
            sim_path = sims_dir / f"{sim.id}.json"
            with open(sim_path, "w") as f:
                f.write(sim.model_dump_json(indent=2))

    def save_metadata(self, path: Path) -> None:
        meta_path, sims_dir = self._resolve_paths(path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        sims_dir.mkdir(parents=True, exist_ok=True)

        if self.simulation_index is None and meta_path.exists():
            with open(meta_path, "r") as f:
                existing = json.loads(f.read())
            existing_index = existing.get("simulation_index")
            if existing_index is not None:
                self.simulation_index = [
                    EvaluationIndexEntry.model_validate(e) for e in existing_index
                ]

        meta = self.model_dump(mode="json", exclude={"simulations"})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_metadata(cls, path: Path) -> "EvaluatedResults":
        path = Path(path)
        fmt = cls._detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                data = json.loads(f.read())
        else:
            meta_path, _ = cls._resolve_paths(path)
            with open(meta_path, "r") as f:
                data = json.loads(f.read())

        data.pop("format_version", None)
        data.pop("simulations", None)
        data["simulations"] = []
        return cls.model_validate(data)

    @classmethod
    def iter_simulations(cls, path: Path) -> Iterator[EvaluatedSimulation]:
        path = Path(path)
        fmt = cls._detect_format(path)
        if fmt == "json":
            with open(path, "r") as f:
                data = json.loads(f.read())
            for sim_data in data.get("simulations", []):
                yield EvaluatedSimulation.model_validate(sim_data)
        else:
            _, sims_dir = cls._resolve_paths(path)
            if sims_dir.exists():
                for sim_file in sorted(sims_dir.glob("*.json")):
                    with open(sim_file, "r") as f:
                        yield EvaluatedSimulation.model_validate_json(f.read())
