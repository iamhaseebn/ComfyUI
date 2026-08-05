from typing import ClassVar

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_audio_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_frame_count,
)


_GENERATE_ENDPOINT = ApiEndpoint(path="/proxy/comfy-cloud/workflow/generate", method="POST")


class _ComfyCloudWorkflowNode(IO.ComfyNode):
    workflow: ClassVar[ComfyCloudWorkflow]
    node_id: ClassVar[str]
    display_name: ClassVar[str]
    category: ClassVar[str]
    requires_image: ClassVar[bool]
    returns_video: ClassVar[bool]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        inputs = [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Describe the content to generate or the edit to apply.",
            )
        ]
        if cls.requires_image:
            inputs.append(IO.Image.Input("image"))

        output = IO.Video.Output() if cls.returns_video else IO.Image.Output()
        return IO.Schema(
            node_id=cls.node_id,
            display_name=cls.display_name,
            category=cls.category,
            inputs=inputs,
            outputs=[output],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(cls, prompt: str, image: Input.Image | None = None) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)

        image_url = None
        if cls.requires_image:
            if get_number_of_images(image) != 1:
                raise ValueError("Exactly one input image is required.")
            image_url = await upload_image_to_comfyapi(cls, image)

        task = await sync_op(
            cls,
            _GENERATE_ENDPOINT,
            response_model=ComfyCloudGenerateResponse,
            data=ComfyCloudGenerateRequest(
                workflow=cls.workflow,
                inputs=ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url),
            ),
        )
        result = await poll_op(
            cls,
            ApiEndpoint(path=task.polling_url),
            response_model=ComfyCloudStatusResponse,
            status_extractor=lambda response: response.status,
            progress_extractor=lambda response: response.progress,
            cancel_endpoint=ApiEndpoint(path=task.cancel_url, method="POST"),
        )
        if not result.output_url:
            detail = f": {result.error}" if result.error else ""
            raise RuntimeError(f"Comfy Cloud task {result.task_id} completed without an output URL{detail}")

        if cls.returns_video:
            output = await download_url_to_video_output(result.output_url, cls=cls)
        else:
            output = await download_url_to_image_tensor(result.output_url, cls=cls)
        return IO.NodeOutput(output)


class ComfyCloudTextToImageNode(_ComfyCloudWorkflowNode):
    workflow = "text-to-image"
    node_id = "ComfyCloudTextToImageNode"
    display_name = "Comfy Cloud Text to Image"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False


class ComfyCloudTextToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "text-to-video"
    node_id = "ComfyCloudTextToVideoNode"
    display_name = "Comfy Cloud Text to Video"
    category = "partner/video/Comfy Cloud"
    requires_image = False
    returns_video = True


class ComfyCloudImageToVideoNode(_ComfyCloudWorkflowNode):
    workflow = "image-to-video"
    node_id = "ComfyCloudImageToVideoNode"
    display_name = "Comfy Cloud Image to Video"
    category = "partner/video/Comfy Cloud"
    requires_image = True
    returns_video = True


class ComfyCloudImageEditNode(_ComfyCloudWorkflowNode):
    workflow = "image-edit"
    node_id = "ComfyCloudImageEditNode"
    display_name = "Comfy Cloud Image Edit"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False


async def _run_video_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
    task = await sync_op(
        cls,
        _GENERATE_ENDPOINT,
        response_model=ComfyCloudGenerateResponse,
        data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs),
    )
    result = await poll_op(
        cls,
        ApiEndpoint(path=task.polling_url),
        response_model=ComfyCloudStatusResponse,
        status_extractor=lambda response: response.status,
        progress_extractor=lambda response: response.progress,
        cancel_endpoint=ApiEndpoint(path=task.cancel_url, method="POST"),
    )
    if not result.output_url:
        detail = f": {result.error}" if result.error else ""
        raise RuntimeError(f"Comfy Cloud task {result.task_id} completed without an output URL{detail}")
    return IO.NodeOutput(await download_url_to_video_output(result.output_url, cls=cls))


def _video_schema(node_id: str, display_name: str, inputs: list) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/video/Comfy Cloud",
        inputs=inputs,
        outputs=[IO.Video.Output()],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
    )


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _seed_input(default: int) -> IO.Int.Input:
    return IO.Int.Input("seed", default=default, min=0, max=18446744073709551615, control_after_generate=True)


class ComfyCloudMiniMaxH3TextSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3TextSoundNode",
            "MiniMax H3 Text + Sound",
            [
                _prompt_input(),
                IO.Combo.Input("aspect_ratio", options=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"], default="1:1"),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(cls, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        return await _run_video_workflow(cls, "video.minimax-h3-text-sound.v1", ComfyCloudWorkflowInputs(prompt=prompt, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudMiniMaxH3ImageSoundNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudMiniMaxH3ImageSoundNode",
            "MiniMax H3 Image + Sound",
            [
                IO.Image.Input("image"),
                _prompt_input(),
                IO.Combo.Input("aspect_ratio", options=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"], default="1:1"),
                IO.Float.Input("duration_seconds", default=5, min=5, max=15, step=0.01),
                _seed_input(168866841893410),
            ],
        )

    @classmethod
    async def execute(cls, image: Input.Image, prompt: str, aspect_ratio: str, duration_seconds: float, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        image_url = await upload_image_to_comfyapi(cls, image)
        return await _run_video_workflow(cls, "video.minimax-h3-image-sound.v1", ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudLTX23ImageAudioPerformanceNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudLTX23ImageAudioPerformanceNode",
            "LTX-2.3 Image + Audio Performance",
            [
                IO.Image.Input("image"), IO.Audio.Input("audio"), _prompt_input(),
                IO.Boolean.Input("enhance_prompt", default=True),
                IO.Float.Input("duration_seconds", default=9, min=1, max=15, step=0.01, tooltip="Must not exceed the input audio duration."),
                _seed_input(225158785956033),
            ],
        )

    @classmethod
    async def execute(cls, image: Input.Image, audio: Input.Audio, prompt: str, enhance_prompt: bool, duration_seconds: float, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        audio_duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        if duration_seconds > audio_duration:
            raise ValueError(f"Duration ({duration_seconds:g}s) exceeds input audio duration ({audio_duration:.2f}s).")
        image_url = await upload_image_to_comfyapi(cls, image)
        audio_url = await upload_audio_to_comfyapi(cls, audio)
        return await _run_video_workflow(cls, "video.ltx-2-3-image-audio-performance.v1", ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url, audio_url=audio_url, enhance_prompt=enhance_prompt, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudLTX23FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudLTX23FirstLastFrameNode",
            "LTX-2.3 First & Last Frame",
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.Int.Input("duration_seconds", default=5, min=2, max=10, step=1, tooltip="25 fps; output frame count is duration × 25 + 1."), _seed_input(315253765879496)],
        )

    @classmethod
    async def execute(cls, first_frame: Input.Image, last_frame: Input.Image, prompt: str, duration_seconds: int, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        if get_number_of_images(first_frame) != 1 or get_number_of_images(last_frame) != 1:
            raise ValueError("Exactly one first frame and one last frame are required.")
        first_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        last_url = await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
        return await _run_video_workflow(cls, "video.ltx-2-3-first-last-frame.v1", ComfyCloudWorkflowInputs(prompt=prompt, first_frame_url=first_url, last_frame_url=last_url, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudWan22FirstLastFrameNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudWan22FirstLastFrameNode",
            "Wan 2.2 14B First & Last Frame",
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.String.Input("negative_prompt", multiline=True, default="graph tested Chinese quality negative"), IO.Int.Input("duration_seconds", default=5, min=2, max=8, step=1, tooltip="Graph frame count is floor(duration × 16 + 1)."), _seed_input(984937593540091)],
        )

    @classmethod
    async def execute(cls, first_frame: Input.Image, last_frame: Input.Image, prompt: str, negative_prompt: str, duration_seconds: int, seed: int) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        validate_string(negative_prompt, min_length=0, max_length=2048)
        if get_number_of_images(first_frame) != 1 or get_number_of_images(last_frame) != 1:
            raise ValueError("Exactly one first frame and one last frame are required.")
        first_url = await upload_image_to_comfyapi(cls, first_frame, wait_label="Uploading first frame")
        last_url = await upload_image_to_comfyapi(cls, last_frame, wait_label="Uploading last frame")
        return await _run_video_workflow(cls, "video.wan-2-2-14b-first-last-frame.v1", ComfyCloudWorkflowInputs(prompt=prompt, negative_prompt=negative_prompt, first_frame_url=first_url, last_frame_url=last_url, duration_seconds=duration_seconds, seed=seed))


class ComfyCloudSCAIL2CharacterReplacementNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _video_schema(
            "ComfyCloudSCAIL2CharacterReplacementNode",
            "SCAIL-2 Character Replacement",
            [IO.Image.Input("reference_character"), IO.Video.Input("driving_video", tooltip="Must contain 81–157 decoded frames."), _prompt_input("scene_prompt"), IO.String.Input("driving_subject", default=""), IO.String.Input("reference_subject", default="human"), _seed_input(1)],
        )

    @classmethod
    async def execute(cls, reference_character: Input.Image, driving_video: Input.Video, scene_prompt: str, driving_subject: str, reference_subject: str, seed: int) -> IO.NodeOutput:
        validate_string(scene_prompt, min_length=1, max_length=4096)
        validate_string(driving_subject, min_length=1, max_length=256)
        validate_string(reference_subject, min_length=1, max_length=256)
        if get_number_of_images(reference_character) != 1:
            raise ValueError("Exactly one reference character image is required.")
        validate_video_frame_count(driving_video, min_frame_count=81, max_frame_count=157)
        image_url = await upload_image_to_comfyapi(cls, reference_character)
        video_url = await upload_video_to_comfyapi(cls, driving_video)
        return await _run_video_workflow(cls, "video.scail-2-character-replacement.v1", ComfyCloudWorkflowInputs(scene_prompt=scene_prompt, driving_subject=driving_subject, reference_subject=reference_subject, reference_character_url=image_url, driving_video_url=video_url, seed=seed))


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ComfyCloudTextToImageNode,
            ComfyCloudTextToVideoNode,
            ComfyCloudImageToVideoNode,
            ComfyCloudImageEditNode,
            ComfyCloudMiniMaxH3TextSoundNode,
            ComfyCloudMiniMaxH3ImageSoundNode,
            ComfyCloudLTX23ImageAudioPerformanceNode,
            ComfyCloudLTX23FirstLastFrameNode,
            ComfyCloudWan22FirstLastFrameNode,
            ComfyCloudSCAIL2CharacterReplacementNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()
