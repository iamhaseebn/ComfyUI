from typing import ClassVar

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudAssetInput,
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_audio_input,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    sync_op,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
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
            image_url = await cls._upload_image(image)

        return await cls._run(ComfyCloudWorkflowInputs(prompt=prompt, image_url=image_url))

    @classmethod
    async def _upload_image(cls, image: Input.Image, total_pixels: int | None = 2048 * 2048) -> str:
        if get_number_of_images(image) != 1:
            raise ValueError("Exactly one input image is required.")
        return await upload_image_to_comfyapi(cls, image, total_pixels=total_pixels)

    @classmethod
    async def _run(cls, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
        task = await sync_op(
            cls,
            _GENERATE_ENDPOINT,
            response_model=ComfyCloudGenerateResponse,
            data=ComfyCloudGenerateRequest(
                workflow=cls.workflow,
                inputs=inputs,
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


_ASPECT_RATIOS = ["1:1", "4:5", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16"]
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def _prompt_input(name: str = "prompt") -> IO.String.Input:
    return IO.String.Input(name, multiline=True, default="")


def _aspect_ratio_input() -> IO.Combo.Input:
    return IO.Combo.Input("aspect_ratio", options=_ASPECT_RATIOS, default="1:1")


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input("seed", default=0, min=0, max=_UINT64_MAX, control_after_generate=True)


def _image_schema(node_id: str, display_name: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/image/Comfy Cloud",
        inputs=inputs,
        outputs=[IO.Image.Output()],
        hidden=[
            IO.Hidden.auth_token_comfy_org,
            IO.Hidden.api_key_comfy_org,
            IO.Hidden.unique_id,
        ],
        is_api_node=True,
    )


class ComfyCloudIdeogram4DesignNode(_ComfyCloudWorkflowNode):
    workflow = "image.ideogram-4-design.v1"
    node_id = "ComfyCloudIdeogram4DesignNode"
    display_name = "Ideogram 4 Design"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                _prompt_input(),
                _aspect_ratio_input(),
                IO.Combo.Input(
                    "quality_mode", options=["quality", "balanced", "fast"], default="balanced"
                ),
                _seed_input(),
            ],
        )

    @classmethod
    async def execute(
        cls, prompt: str, aspect_ratio: str = "1:1", quality_mode: str = "balanced", seed: int = 0
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, aspect_ratio=aspect_ratio, quality_mode=quality_mode, seed=seed
            )
        )


class ComfyCloudKrea2CreativeImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.krea-2-creative-image.v1"
    node_id = "ComfyCloudKrea2CreativeImageNode"
    display_name = "Krea 2 Creative Image"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                _prompt_input(),
                IO.Boolean.Input("prompt_enhance", default=True),
                _aspect_ratio_input(),
                _seed_input(),
            ],
        )

    @classmethod
    async def execute(
        cls, prompt: str, prompt_enhance: bool = True, aspect_ratio: str = "1:1", seed: int = 0
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, prompt_enhance=prompt_enhance, aspect_ratio=aspect_ratio, seed=seed
            )
        )


class ComfyCloudMageFlowImageNode(_ComfyCloudWorkflowNode):
    workflow = "image.mage-flow-image.v1"
    node_id = "ComfyCloudMageFlowImageNode"
    display_name = "Mage-Flow Image"
    category = "partner/image/Comfy Cloud"
    requires_image = False
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                _prompt_input(),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                _aspect_ratio_input(),
                _seed_input(),
            ],
        )

    @classmethod
    async def execute(
        cls, prompt: str, negative_prompt: str = "", aspect_ratio: str = "1:1", seed: int = 0
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        validate_string(negative_prompt, min_length=0, max_length=2048, field_name="negative_prompt")
        return await cls._run(
            ComfyCloudWorkflowInputs(
                prompt=prompt, negative_prompt=negative_prompt, aspect_ratio=aspect_ratio, seed=seed
            )
        )


class ComfyCloudFlux2ReferenceEditNode(_ComfyCloudWorkflowNode):
    workflow = "image.flux-2-reference-edit.v1"
    node_id = "ComfyCloudFlux2ReferenceEditNode"
    display_name = "FLUX.2 Reference Edit"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Float.Input("guidance", default=4.0, min=1.0, max=10.0, step=0.1),
                IO.Combo.Input("quality_mode", options=["quality", "fast"], default="quality"),
                _seed_input(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        guidance: float = 4.0,
        quality_mode: str = "quality",
        seed: int = 0,
    ) -> IO.NodeOutput:
        validate_string(instruction, min_length=1, max_length=4096, field_name="instruction")
        return await cls._run(
            ComfyCloudWorkflowInputs(
                assets={
                    "image": ComfyCloudAssetInput(
                        type="IMAGE", url=await cls._upload_image(image, total_pixels=None)
                    )
                },
                instruction=instruction,
                guidance=guidance,
                quality_mode=quality_mode,
                seed=seed,
            )
        )


class ComfyCloudQwenImageEdit2511Node(_ComfyCloudWorkflowNode):
    workflow = "image.qwen-image-edit-2511.v1"
    node_id = "ComfyCloudQwenImageEdit2511Node"
    display_name = "Qwen Image Edit 2511"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [
                IO.Image.Input("image"),
                _prompt_input("instruction"),
                IO.Combo.Input("quality_mode", options=["quality", "fast"], default="quality"),
                _seed_input(),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        quality_mode: str = "quality",
        seed: int = 0,
    ) -> IO.NodeOutput:
        validate_string(instruction, min_length=1, max_length=4096, field_name="instruction")
        return await cls._run(
            ComfyCloudWorkflowInputs(
                assets={
                    "image": ComfyCloudAssetInput(
                        type="IMAGE", url=await cls._upload_image(image, total_pixels=None)
                    )
                },
                instruction=instruction,
                quality_mode=quality_mode,
                seed=seed,
            )
        )


class ComfyCloudSeedVR2ImageUpscaleNode(_ComfyCloudWorkflowNode):
    workflow = "image.seedvr2-image-upscale.v1"
    node_id = "ComfyCloudSeedVR2ImageUpscaleNode"
    display_name = "SeedVR2 Image Upscale"
    category = "partner/image/Comfy Cloud"
    requires_image = True
    returns_video = False

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _image_schema(
            cls.node_id,
            cls.display_name,
            [IO.Image.Input("image"), IO.Combo.Input("scale", options=["2x", "4x"], default="4x")],
        )

    @classmethod
    async def execute(cls, image: Input.Image, scale: str = "4x") -> IO.NodeOutput:
        return await cls._run(
            ComfyCloudWorkflowInputs(
                assets={
                    "image": ComfyCloudAssetInput(
                        type="IMAGE", url=await cls._upload_image(image, total_pixels=None)
                    )
                },
                scale=scale,
            )
        )


async def _run_video_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs) -> IO.NodeOutput:
    task = await sync_op(cls, _GENERATE_ENDPOINT, response_model=ComfyCloudGenerateResponse, data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs))
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


def _video_schema(node_id: str, display_name: str, inputs: list[IO.Input]) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/video/Comfy Cloud",
        inputs=inputs,
        outputs=[IO.Video.Output()],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
    )


def _video_seed_input(default: int) -> IO.Int.Input:
    return IO.Int.Input("seed", default=default, min=0, max=_UINT64_MAX, control_after_generate=True)


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
                _video_seed_input(168866841893410),
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
                _video_seed_input(168866841893410),
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
                _video_seed_input(225158785956033),
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
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.Int.Input("duration_seconds", default=5, min=2, max=10, step=1, tooltip="25 fps; output frame count is duration × 25 + 1."), _video_seed_input(315253765879496)],
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
            [IO.Image.Input("first_frame"), IO.Image.Input("last_frame"), _prompt_input(), IO.String.Input("negative_prompt", multiline=True, default="graph tested Chinese quality negative"), IO.Int.Input("duration_seconds", default=5, min=2, max=8, step=1, tooltip="Graph frame count is floor(duration × 16 + 1)."), _video_seed_input(984937593540091)],
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
            [IO.Image.Input("reference_character"), IO.Video.Input("driving_video", tooltip="Must contain 81–157 decoded frames."), _prompt_input("scene_prompt"), IO.String.Input("driving_subject", default=""), IO.String.Input("reference_subject", default="human"), _video_seed_input(1)],
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


_UINT32_MAX = 0xFFFFFFFF
_ACE_LANGUAGES = ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown"]
_ACE_KEYS = [f"{root} {mode}" for mode in ("major", "minor") for root in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")]
_CHATTERBOX_LANGUAGES = ["Arabic (ar)", "Danish (da)", "German (de)", "Greek (el)", "English (en)", "Spanish (es)", "Finnish (fi)", "French (fr)", "Hebrew (he)", "Hindi (hi)", "Italian (it)", "Japanese (ja)", "Korean (ko)", "Malay (ms)", "Dutch (nl)", "Norwegian (no)", "Polish (pl)", "Portuguese (pt)", "Russian (ru)", "Swedish (sv)", "Swahili (sw)", "Turkish (tr)", "Chinese (zh)"]


def _audio_schema(node_id: str, display_name: str, inputs: list[IO.Input], outputs: list[IO.Output] | None = None) -> IO.Schema:
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category="partner/audio/Comfy Cloud",
        inputs=inputs,
        outputs=outputs or [IO.Audio.Output()],
        hidden=[IO.Hidden.auth_token_comfy_org, IO.Hidden.api_key_comfy_org, IO.Hidden.unique_id],
        is_api_node=True,
    )


def _audio_duration(audio: Input.Audio) -> float:
    return audio["waveform"].shape[-1] / audio["sample_rate"]


def _validate_audio_duration(name: str, audio: Input.Audio, minimum: float, maximum: float) -> None:
    duration = _audio_duration(audio)
    if duration < minimum or duration > maximum:
        raise ValueError(f"{name} duration must be between {minimum:g} and {maximum:g} seconds.")


async def _audio_asset(cls: type[IO.ComfyNode], name: str, audio: Input.Audio) -> dict[str, ComfyCloudAssetInput]:
    return {name: ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, audio))}


async def _run_audio_workflow(cls: type[IO.ComfyNode], workflow: ComfyCloudWorkflow, inputs: ComfyCloudWorkflowInputs, output_names: tuple[str, ...] = ()) -> IO.NodeOutput:
    task = await sync_op(cls, _GENERATE_ENDPOINT, response_model=ComfyCloudGenerateResponse, data=ComfyCloudGenerateRequest(workflow=workflow, inputs=inputs))
    result = await poll_op(
        cls,
        ApiEndpoint(path=task.polling_url),
        response_model=ComfyCloudStatusResponse,
        status_extractor=lambda response: response.status,
        progress_extractor=lambda response: response.progress,
        cancel_endpoint=ApiEndpoint(path=task.cancel_url, method="POST"),
    )
    if output_names:
        if not result.output_urls or any(not result.output_urls.get(name) for name in output_names):
            detail = f": {result.error}" if result.error else ""
            raise RuntimeError(f"Comfy Cloud task {result.task_id} completed without all named output URLs{detail}")
        outputs = [await download_url_to_audio_input(result.output_urls[name], cls=cls) for name in output_names]
        return IO.NodeOutput(*outputs)
    if not result.output_url:
        detail = f": {result.error}" if result.error else ""
        raise RuntimeError(f"Comfy Cloud task {result.task_id} completed without an output URL{detail}")
    return IO.NodeOutput(await download_url_to_audio_input(result.output_url, cls=cls))


class ComfyCloudACEStep15XLTurboNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudACEStep15XLTurboNode",
            "ACE-Step 1.5 XL Turbo",
            [
                _prompt_input("style_prompt"),
                IO.String.Input("lyrics", multiline=True, default=""),
                IO.Float.Input("duration_seconds", default=120, min=10, max=300, step=0.1),
                _seed_input(),
                IO.Int.Input("bpm", default=120, min=10, max=300),
                IO.Combo.Input("time_signature", options=["2", "3", "4", "6"], default="4"),
                IO.Combo.Input("language", options=_ACE_LANGUAGES, default="en"),
                IO.Combo.Input("key", options=_ACE_KEYS, default="E minor"),
            ],
        )

    @classmethod
    async def execute(cls, style_prompt: str, lyrics: str, duration_seconds: float, seed: int, bpm: int, time_signature: str, language: str, key: str) -> IO.NodeOutput:
        validate_string(style_prompt, min_length=1, max_length=4096, field_name="style_prompt")
        validate_string(lyrics, min_length=0, max_length=20000, field_name="lyrics")
        return await _run_audio_workflow(cls, "audio.ace-step-1-5-xl-turbo.v1", ComfyCloudWorkflowInputs(style_prompt=style_prompt, lyrics=lyrics, duration_seconds=duration_seconds, seed=seed, bpm=bpm, time_signature=time_signature, language=language, key=key))


class ComfyCloudStableAudio3MediumNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudStableAudio3MediumNode",
            "Stable Audio 3 Medium",
            [
                _prompt_input(),
                IO.Float.Input("duration_seconds", default=30, min=1, max=300, step=0.1),
                _seed_input(),
                IO.Boolean.Input("expand_prompt", default=True),
                IO.Combo.Input("category", options=["Music", "Instrument", "SFX", "One-shot"], default="Music"),
            ],
        )

    @classmethod
    async def execute(cls, prompt: str, duration_seconds: float, seed: int, expand_prompt: bool, category: str) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=4096)
        return await _run_audio_workflow(cls, "audio.stable-audio-3-medium.v1", ComfyCloudWorkflowInputs(prompt=prompt, duration_seconds=duration_seconds, seed=seed, expand_prompt=expand_prompt, category=category))


class ComfyCloudChatterboxMultilingualVoiceCloneNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxMultilingualVoiceCloneNode",
            "Chatterbox Multilingual Voice Clone",
            [
                _prompt_input("text"), IO.Audio.Input("voice_reference"),
                IO.Combo.Input("language", options=_CHATTERBOX_LANGUAGES, default="English (en)"),
                IO.Float.Input("exaggeration", default=0.5, min=0, max=2, step=0.05),
                IO.Float.Input("cfg_weight", default=0.5, min=0, max=1, step=0.05),
                IO.Float.Input("temperature", default=0.8, min=0.05, max=2, step=0.05),
                IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True),
            ],
        )

    @classmethod
    async def execute(cls, text: str, voice_reference: Input.Audio, language: str, exaggeration: float, cfg_weight: float, temperature: float, seed: int) -> IO.NodeOutput:
        validate_string(text, min_length=1, max_length=5000, field_name="text")
        _validate_audio_duration("Voice reference", voice_reference, 1, 30)
        return await _run_audio_workflow(cls, "audio.chatterbox-multilingual-voice-clone.v1", ComfyCloudWorkflowInputs(text=text, assets=await _audio_asset(cls, "voice_reference", voice_reference), language=language, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature, seed=seed))


class ComfyCloudChatterboxDialogueNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxDialogueNode",
            "Chatterbox Dialogue",
            [
                _prompt_input("script"), IO.Audio.Input("speaker_a_reference"), IO.Audio.Input("speaker_b_reference"),
                IO.Float.Input("exaggeration", default=0.5, min=0.25, max=2, step=0.05),
                IO.Float.Input("cfg_weight", default=0.5, min=0.2, max=1, step=0.05),
                IO.Float.Input("temperature", default=0.8, min=0.05, max=5, step=0.05),
                IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True),
            ],
        )

    @classmethod
    async def execute(cls, script: str, speaker_a_reference: Input.Audio, speaker_b_reference: Input.Audio, exaggeration: float, cfg_weight: float, temperature: float, seed: int) -> IO.NodeOutput:
        validate_string(script, min_length=1, max_length=10000, field_name="script")
        if any(line.strip() and not line.strip().startswith(("SPEAKER A:", "SPEAKER B:", "SPEAKER C:", "SPEAKER D:")) for line in script.splitlines()):
            raise ValueError("Every nonblank utterance must start with SPEAKER A: through SPEAKER D:.")
        _validate_audio_duration("Speaker A reference", speaker_a_reference, 1, 30)
        _validate_audio_duration("Speaker B reference", speaker_b_reference, 1, 30)
        assets = {
            "speaker_a_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, speaker_a_reference)),
            "speaker_b_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, speaker_b_reference)),
        }
        return await _run_audio_workflow(cls, "audio.chatterbox-dialogue.v1", ComfyCloudWorkflowInputs(script=script, assets=assets, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature, seed=seed))


class ComfyCloudChatterboxVoiceConversionNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudChatterboxVoiceConversionNode",
            "Chatterbox Voice Conversion",
            [IO.Audio.Input("source_audio"), IO.Audio.Input("target_voice_reference"), IO.Int.Input("seed", default=0, min=0, max=_UINT32_MAX, control_after_generate=True)],
        )

    @classmethod
    async def execute(cls, source_audio: Input.Audio, target_voice_reference: Input.Audio, seed: int) -> IO.NodeOutput:
        _validate_audio_duration("Source audio", source_audio, 0.5, 300)
        _validate_audio_duration("Target voice reference", target_voice_reference, 1, 30)
        assets = {
            "source_audio": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, source_audio)),
            "target_voice_reference": ComfyCloudAssetInput(type="AUDIO", url=await upload_audio_to_comfyapi(cls, target_voice_reference)),
        }
        return await _run_audio_workflow(cls, "audio.chatterbox-voice-conversion.v1", ComfyCloudWorkflowInputs(assets=assets, seed=seed))


class ComfyCloudMelBandRoFormerStemSeparationNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return _audio_schema(
            "ComfyCloudMelBandRoFormerStemSeparationNode",
            "MelBandRoFormer Stem Separation",
            [IO.Audio.Input("audio")],
            [IO.Audio.Output("vocals"), IO.Audio.Output("instruments")],
        )

    @classmethod
    async def execute(cls, audio: Input.Audio) -> IO.NodeOutput:
        _validate_audio_duration("Audio", audio, 0.5, 600)
        return await _run_audio_workflow(cls, "audio.melbandroformer-stem-separation.v1", ComfyCloudWorkflowInputs(assets=await _audio_asset(cls, "audio", audio)), ("vocals", "instruments"))


class ComfyCloudExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ComfyCloudTextToImageNode,
            ComfyCloudTextToVideoNode,
            ComfyCloudImageToVideoNode,
            ComfyCloudImageEditNode,
            ComfyCloudIdeogram4DesignNode,
            ComfyCloudKrea2CreativeImageNode,
            ComfyCloudMageFlowImageNode,
            ComfyCloudFlux2ReferenceEditNode,
            ComfyCloudQwenImageEdit2511Node,
            ComfyCloudSeedVR2ImageUpscaleNode,
            ComfyCloudMiniMaxH3TextSoundNode,
            ComfyCloudMiniMaxH3ImageSoundNode,
            ComfyCloudLTX23ImageAudioPerformanceNode,
            ComfyCloudLTX23FirstLastFrameNode,
            ComfyCloudWan22FirstLastFrameNode,
            ComfyCloudSCAIL2CharacterReplacementNode,
            ComfyCloudACEStep15XLTurboNode,
            ComfyCloudStableAudio3MediumNode,
            ComfyCloudChatterboxMultilingualVoiceCloneNode,
            ComfyCloudChatterboxDialogueNode,
            ComfyCloudChatterboxVoiceConversionNode,
            ComfyCloudMelBandRoFormerStemSeparationNode,
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()
