from typing import Literal

from pydantic import BaseModel, Field


ComfyCloudWorkflow = Literal[
    "text-to-image",
    "text-to-video",
    "image-to-video",
    "image-edit",
    "image.ideogram-4-design.v1",
    "image.krea-2-creative-image.v1",
    "image.mage-flow-image.v1",
    "image.flux-2-reference-edit.v1",
    "image.qwen-image-edit-2511.v1",
    "image.seedvr2-image-upscale.v1",
]


class ComfyCloudWorkflowInputs(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    instruction: str | None = Field(None)
    prompt_enhance: bool | None = Field(None)
    negative_prompt: str | None = Field(None)
    aspect_ratio: str | None = Field(None)
    guidance: float | None = Field(None)
    quality_mode: str | None = Field(None)
    seed: int | None = Field(None, ge=0, le=0xFFFFFFFFFFFFFFFF)
    scale: str | None = Field(None)


class ComfyCloudGenerateRequest(BaseModel):
    workflow: ComfyCloudWorkflow = Field(...)
    inputs: ComfyCloudWorkflowInputs = Field(...)


class ComfyCloudGenerateResponse(BaseModel):
    task_id: str = Field(...)
    status: str = Field(...)
    polling_url: str = Field(...)
    cancel_url: str = Field(...)


class ComfyCloudStatusResponse(BaseModel):
    task_id: str = Field(...)
    status: str = Field(...)
    progress: float | None = Field(None)
    output_url: str | None = Field(None)
    error: str | None = Field(None)
