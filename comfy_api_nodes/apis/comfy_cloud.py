from typing import Literal

from pydantic import BaseModel, Field


ComfyCloudWorkflow = Literal[
    "text-to-image",
    "text-to-video",
    "image-to-video",
    "image-edit",
    "video.minimax-h3-text-sound.v1",
    "video.minimax-h3-image-sound.v1",
    "video.ltx-2-3-image-audio-performance.v1",
    "video.ltx-2-3-first-last-frame.v1",
    "video.wan-2-2-14b-first-last-frame.v1",
    "video.scail-2-character-replacement.v1",
]


class ComfyCloudWorkflowInputs(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    audio_url: str | None = Field(None)
    first_frame_url: str | None = Field(None)
    last_frame_url: str | None = Field(None)
    reference_character_url: str | None = Field(None)
    driving_video_url: str | None = Field(None)
    aspect_ratio: str | None = Field(None)
    duration_seconds: float | None = Field(None)
    seed: int | None = Field(None, ge=0, le=18446744073709551615)
    enhance_prompt: bool | None = Field(None)
    negative_prompt: str | None = Field(None)
    scene_prompt: str | None = Field(None)
    driving_subject: str | None = Field(None)
    reference_subject: str | None = Field(None)


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
