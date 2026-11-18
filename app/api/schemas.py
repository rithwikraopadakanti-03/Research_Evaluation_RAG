from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.evaluation.pipelines import GridSpec, PipelineConfig


class BenchmarkRequest(BaseModel):
    dataset_id: str
    pipelines: Optional[list[PipelineConfig]] = None
    grid: Optional[GridSpec] = None
    metric: str = "mrr"
    use_judge: bool = False

    @model_validator(mode="after")
    def _one_source(self):
        if self.pipelines and self.grid:
            raise ValueError("Provide either pipelines or grid, not both")
        if not self.pipelines and not self.grid:
            self.grid = GridSpec()  # default: dense vs bm25 vs hybrid
        return self
