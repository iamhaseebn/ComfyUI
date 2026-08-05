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
    validate_string,
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
                image_url=await cls._upload_image(image, total_pixels=None),
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
                image_url=await cls._upload_image(image, total_pixels=None),
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
            ComfyCloudWorkflowInputs(image_url=await cls._upload_image(image, total_pixels=None), scale=scale)
        )


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
        ]


async def comfy_entrypoint() -> ComfyCloudExtension:
    return ComfyCloudExtension()
