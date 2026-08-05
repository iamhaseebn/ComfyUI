import asyncio
from io import BytesIO
from typing import get_args
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_api_nodes.apis.comfy_cloud import (
    ComfyCloudGenerateRequest,
    ComfyCloudGenerateResponse,
    ComfyCloudStatusResponse,
    ComfyCloudWorkflow,
    ComfyCloudWorkflowInputs,
)
from comfy_api_nodes import nodes_comfy_cloud
from comfy_api_nodes.util import download_helpers


@pytest.mark.parametrize(
    ("node", "workflow", "returns_video", "requires_image"),
    [
        (nodes_comfy_cloud.ComfyCloudTextToImageNode, "text-to-image", False, False),
        (nodes_comfy_cloud.ComfyCloudTextToVideoNode, "text-to-video", True, False),
        (nodes_comfy_cloud.ComfyCloudImageToVideoNode, "image-to-video", True, True),
        (nodes_comfy_cloud.ComfyCloudImageEditNode, "image-edit", False, True),
    ],
)
def test_workflow_submission_polling_and_download(monkeypatch, node, workflow, returns_video, requires_image):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-1",
            status="queued",
            polling_url="/proxy/comfy-cloud/workflow/tasks/task-1",
            cancel_url="/proxy/comfy-cloud/workflow/tasks/task-1/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-1",
            status="completed",
            progress=100,
            output_url="https://example.com/output",
        )
    )
    upload = AsyncMock(return_value="https://example.com/input.png")
    image_download = AsyncMock(return_value="image-output")
    video_download = AsyncMock(return_value="video-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", image_download)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_video_output", video_download)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    image = object() if requires_image else None
    output = asyncio.run(node.execute("A tiny fennec fox", image))

    endpoint = sync.call_args.args[1]
    request = sync.call_args.kwargs["data"]
    assert endpoint.path == "/proxy/comfy-cloud/workflow/generate"
    assert endpoint.method == "POST"
    assert request == ComfyCloudGenerateRequest(
        workflow=workflow,
        inputs=ComfyCloudWorkflowInputs(
            prompt="A tiny fennec fox",
            image_url="https://example.com/input.png" if requires_image else None,
        ),
    )
    assert upload.await_count == int(requires_image)

    poll_endpoint = poll.call_args.args[1]
    cancel_endpoint = poll.call_args.kwargs["cancel_endpoint"]
    assert poll_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1"
    assert cancel_endpoint.path == "/proxy/comfy-cloud/workflow/tasks/task-1/cancel"
    assert cancel_endpoint.method == "POST"
    assert output[0] == ("video-output" if returns_video else "image-output")


@pytest.mark.parametrize(
    "node",
    [nodes_comfy_cloud.ComfyCloudImageToVideoNode, nodes_comfy_cloud.ComfyCloudImageEditNode],
)
def test_image_workflows_reject_batches(monkeypatch, node):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 2)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)

    with pytest.raises(ValueError, match="Exactly one input image"):
        asyncio.run(node.execute("Animate this", object()))
    upload.assert_not_awaited()


def test_contract_omits_optional_status_fields():
    request = ComfyCloudGenerateRequest(
        workflow="text-to-image",
        inputs=ComfyCloudWorkflowInputs(prompt="A lighthouse"),
    )
    status = ComfyCloudStatusResponse(task_id="task-1", status="queued")

    assert request.model_dump(exclude_none=True) == {
        "workflow": "text-to-image",
        "inputs": {"prompt": "A lighthouse"},
    }
    assert status.model_dump(exclude_none=True) == {"task_id": "task-1", "status": "queued"}


IMAGE_POC_NODES = [
    (
        nodes_comfy_cloud.ComfyCloudIdeogram4DesignNode,
        "image.ideogram-4-design.v1",
        ["prompt", "aspect_ratio", "quality_mode", "seed"],
        {
            "prompt": "A geometric fox logo",
            "aspect_ratio": "4:5",
            "quality_mode": "fast",
            "seed": 11,
        },
    ),
    (
        nodes_comfy_cloud.ComfyCloudKrea2CreativeImageNode,
        "image.krea-2-creative-image.v1",
        ["prompt", "prompt_enhance", "aspect_ratio", "seed"],
        {"prompt": "A glass forest", "prompt_enhance": False, "aspect_ratio": "16:9", "seed": 12},
    ),
    (
        nodes_comfy_cloud.ComfyCloudMageFlowImageNode,
        "image.mage-flow-image.v1",
        ["prompt", "negative_prompt", "aspect_ratio", "seed"],
        {"prompt": "A moonlit lake", "negative_prompt": "fog", "aspect_ratio": "3:2", "seed": 13},
    ),
    (
        nodes_comfy_cloud.ComfyCloudFlux2ReferenceEditNode,
        "image.flux-2-reference-edit.v1",
        ["image", "instruction", "guidance", "quality_mode", "seed"],
        {"image": object(), "instruction": "Make it winter", "guidance": 5.5, "quality_mode": "fast", "seed": 14},
    ),
    (
        nodes_comfy_cloud.ComfyCloudQwenImageEdit2511Node,
        "image.qwen-image-edit-2511.v1",
        ["image", "instruction", "quality_mode", "seed"],
        {"image": object(), "instruction": "Remove the sign", "quality_mode": "fast", "seed": 15},
    ),
    (
        nodes_comfy_cloud.ComfyCloudSeedVR2ImageUpscaleNode,
        "image.seedvr2-image-upscale.v1",
        ["image", "scale"],
        {"image": object(), "scale": "2x"},
    ),
]


@pytest.mark.parametrize(("node", "workflow", "input_names", "arguments"), IMAGE_POC_NODES)
def test_image_poc_node_schema_and_request_mapping(monkeypatch, node, workflow, input_names, arguments):
    sync = AsyncMock(
        return_value=ComfyCloudGenerateResponse(
            task_id="task-poc",
            status="queued",
            polling_url="/tasks/task-poc",
            cancel_url="/tasks/task-poc/cancel",
        )
    )
    poll = AsyncMock(
        return_value=ComfyCloudStatusResponse(
            task_id="task-poc", status="completed", output_url="/results/task-poc/image.png"
        )
    )
    upload = AsyncMock(return_value="/uploads/input.png")
    download = AsyncMock(return_value="image-output")
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_image_tensor", download)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    schema = node.define_schema()
    assert schema.node_id == node.node_id
    assert schema.display_name == node.display_name
    assert schema.is_api_node is True
    assert [input.id for input in schema.inputs] == input_names
    assert len(schema.outputs) == 1
    assert schema.outputs[0].get_io_type() == "IMAGE"

    output = asyncio.run(node.execute(**arguments))
    request = sync.call_args.kwargs["data"]
    expected_inputs = {key: value for key, value in arguments.items() if key != "image"}
    if "image" in arguments:
        expected_inputs["assets"] = {"image": {"type": "IMAGE", "url": "/uploads/input.png"}}

    assert request.workflow == workflow
    assert request.inputs.model_dump(exclude_none=True) == expected_inputs
    assert "asset_id" not in request.model_dump_json()
    assert '"id"' not in request.model_dump_json()
    assert upload.await_count == int("image" in arguments)
    if "image" in arguments:
        assert upload.call_args.kwargs == {"total_pixels": None}
    download.assert_awaited_once_with("/results/task-poc/image.png", cls=node)
    assert output[0] == "image-output"


def test_image_poc_schema_defaults_ranges_and_enums():
    schemas = {
        node.workflow: {input.id: input for input in node.define_schema().inputs}
        for node, _, _, _ in IMAGE_POC_NODES
    }
    aspect_ratios = ["1:1", "4:5", "3:4", "2:3", "3:2", "4:3", "16:9", "9:16"]

    for workflow in [
        "image.ideogram-4-design.v1",
        "image.krea-2-creative-image.v1",
        "image.mage-flow-image.v1",
    ]:
        assert schemas[workflow]["aspect_ratio"].options == aspect_ratios
        assert schemas[workflow]["aspect_ratio"].default == "1:1"
        seed = schemas[workflow]["seed"]
        assert (seed.default, seed.min, seed.max) == (0, 0, 0xFFFFFFFFFFFFFFFF)

    assert schemas["image.ideogram-4-design.v1"]["quality_mode"].options == ["quality", "balanced", "fast"]
    assert schemas["image.ideogram-4-design.v1"]["quality_mode"].default == "balanced"
    assert schemas["image.krea-2-creative-image.v1"]["prompt_enhance"].default is True
    assert schemas["image.mage-flow-image.v1"]["negative_prompt"].default == ""

    guidance = schemas["image.flux-2-reference-edit.v1"]["guidance"]
    assert (guidance.default, guidance.min, guidance.max, guidance.step) == (4.0, 1.0, 10.0, 0.1)
    for workflow in ["image.flux-2-reference-edit.v1", "image.qwen-image-edit-2511.v1"]:
        assert schemas[workflow]["quality_mode"].options == ["quality", "fast"]
        assert schemas[workflow]["quality_mode"].default == "quality"
        seed = schemas[workflow]["seed"]
        assert (seed.default, seed.min, seed.max) == (0, 0, 0xFFFFFFFFFFFFFFFF)

    scale = schemas["image.seedvr2-image-upscale.v1"]["scale"]
    assert scale.options == ["2x", "4x"]
    assert scale.default == "4x"


def test_image_poc_api_declarations_and_extension_registration():
    workflows = {workflow for _, workflow, _, _ in IMAGE_POC_NODES}
    registered = set(asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list()))

    assert workflows <= set(get_args(ComfyCloudWorkflow))
    assert {node for node, _, _, _ in IMAGE_POC_NODES} <= registered


@pytest.mark.parametrize(
    ("node", "input_names"),
    [
        (nodes_comfy_cloud.ComfyCloudMiniMaxH3TextSoundNode, ["prompt", "aspect_ratio", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudMiniMaxH3ImageSoundNode, ["image", "prompt", "aspect_ratio", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudLTX23ImageAudioPerformanceNode, ["image", "audio", "prompt", "enhance_prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudLTX23FirstLastFrameNode, ["first_frame", "last_frame", "prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudWan22FirstLastFrameNode, ["first_frame", "last_frame", "prompt", "negative_prompt", "duration_seconds", "seed"]),
        (nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode, ["reference_character", "driving_video", "scene_prompt", "driving_subject", "reference_subject", "seed"]),
    ],
)
def test_video_node_schemas_expose_only_manifest_inputs(node, input_names):
    schema = node.define_schema()
    assert schema.is_api_node
    assert [input.id for input in schema.inputs] == input_names
    assert len(schema.outputs) == 1
    assert schema.outputs[0].get_io_type() == "VIDEO"


def test_ltx_performance_stages_image_and_audio(monkeypatch):
    run = AsyncMock(return_value=("video-output",))
    image_upload = AsyncMock(return_value="https://example.com/image.png")
    audio_upload = AsyncMock(return_value="https://example.com/audio.mp4")
    monkeypatch.setattr(nodes_comfy_cloud, "_run_video_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", image_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", audio_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)
    audio = {"waveform": torch.zeros(1, 1, 480000), "sample_rate": 48000}

    asyncio.run(nodes_comfy_cloud.ComfyCloudLTX23ImageAudioPerformanceNode.execute(object(), audio, "sing", True, 9, 7))

    inputs = run.call_args.args[2]
    assert inputs.image_url == "https://example.com/image.png"
    assert inputs.audio_url == "https://example.com/audio.mp4"
    assert inputs.duration_seconds == 9


def test_scail_stages_reference_image_and_driving_video(monkeypatch):
    run = AsyncMock(return_value=("video-output",))
    image_upload = AsyncMock(return_value="https://example.com/character.png")
    video_upload = AsyncMock(return_value="https://example.com/driving.mp4")
    video = Mock()
    video.get_frame_count.return_value = 100
    monkeypatch.setattr(nodes_comfy_cloud, "_run_video_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_image_to_comfyapi", image_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_video_to_comfyapi", video_upload)
    monkeypatch.setattr(nodes_comfy_cloud, "get_number_of_images", lambda image: 1)

    asyncio.run(nodes_comfy_cloud.ComfyCloudSCAIL2CharacterReplacementNode.execute(object(), video, "park", "woman", "human", 1))

    inputs = run.call_args.args[2]
    assert inputs.reference_character_url == "https://example.com/character.png"
    assert inputs.driving_video_url == "https://example.com/driving.mp4"
    video.get_frame_count.assert_called_once()


def test_download_cloud_audio_url_to_audio_input(monkeypatch):
    node = nodes_comfy_cloud.ComfyCloudTextToImageNode
    downloaded = b"encoded audio"
    expected = {"waveform": torch.ones(1, 2, 3), "sample_rate": 48000}
    download_call = Mock()

    async def download(url, dest, **kwargs):
        download_call(url=url, dest=dest, **kwargs)
        dest.write(downloaded)
        dest.seek(0)

    audio_decode = Mock(return_value=expected)
    monkeypatch.setattr(download_helpers, "download_url_to_bytesio", download)
    monkeypatch.setattr(download_helpers, "audio_bytes_to_audio_input", audio_decode)

    output = asyncio.run(
        download_helpers.download_url_to_audio_input(
            "/proxy/comfy-cloud/results/task-1/audio.flac",
            timeout=30,
            max_retries=2,
            cls=node,
        )
    )

    assert output is expected
    download_call.assert_called_once()
    assert download_call.call_args.kwargs["url"] == "/proxy/comfy-cloud/results/task-1/audio.flac"
    assert isinstance(download_call.call_args.kwargs["dest"], BytesIO)
    assert download_call.call_args.kwargs["timeout"] == 30
    assert download_call.call_args.kwargs["max_retries"] == 2
    assert download_call.call_args.kwargs["cls"] is node
    audio_decode.assert_called_once_with(downloaded)


AUDIO_POC_NODES = [
    (nodes_comfy_cloud.ComfyCloudACEStep15XLTurboNode, "audio.ace-step-1-5-xl-turbo.v1", ["style_prompt", "lyrics", "duration_seconds", "seed", "bpm", "time_signature", "language", "key"]),
    (nodes_comfy_cloud.ComfyCloudStableAudio3MediumNode, "audio.stable-audio-3-medium.v1", ["prompt", "duration_seconds", "seed", "expand_prompt", "category"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxMultilingualVoiceCloneNode, "audio.chatterbox-multilingual-voice-clone.v1", ["text", "voice_reference", "language", "exaggeration", "cfg_weight", "temperature", "seed"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode, "audio.chatterbox-dialogue.v1", ["script", "speaker_a_reference", "speaker_b_reference", "exaggeration", "cfg_weight", "temperature", "seed"]),
    (nodes_comfy_cloud.ComfyCloudChatterboxVoiceConversionNode, "audio.chatterbox-voice-conversion.v1", ["source_audio", "target_voice_reference", "seed"]),
    (nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode, "audio.melbandroformer-stem-separation.v1", ["audio"]),
]


@pytest.mark.parametrize(("node", "workflow", "input_names"), AUDIO_POC_NODES)
def test_audio_poc_schemas_and_registration(node, workflow, input_names):
    schema = node.define_schema()
    registered = asyncio.run(nodes_comfy_cloud.ComfyCloudExtension().get_node_list())

    assert schema.is_api_node
    assert schema.category == "partner/audio/Comfy Cloud"
    assert [input.id for input in schema.inputs] == input_names
    assert all(output.get_io_type() == "AUDIO" for output in schema.outputs)
    assert workflow in get_args(ComfyCloudWorkflow)
    assert node in registered


def test_audio_poc_schema_defaults_ranges_and_enums():
    schemas = {workflow: {input.id: input for input in node.define_schema().inputs} for node, workflow, _ in AUDIO_POC_NODES}
    ace = schemas["audio.ace-step-1-5-xl-turbo.v1"]
    assert (ace["duration_seconds"].default, ace["duration_seconds"].min, ace["duration_seconds"].max, ace["duration_seconds"].step) == (120, 10, 300, 0.1)
    assert (ace["bpm"].default, ace["bpm"].min, ace["bpm"].max) == (120, 10, 300)
    assert ace["time_signature"].options == ["2", "3", "4", "6"]
    assert ace["language"].default == "en"
    assert ace["key"].default == "E minor"

    stable = schemas["audio.stable-audio-3-medium.v1"]
    assert stable["category"].options == ["Music", "Instrument", "SFX", "One-shot"]
    assert stable["expand_prompt"].default is True
    for workflow in [
        "audio.chatterbox-multilingual-voice-clone.v1",
        "audio.chatterbox-dialogue.v1",
        "audio.chatterbox-voice-conversion.v1",
    ]:
        assert schemas[workflow]["seed"].max == 0xFFFFFFFF

    mel_schema = nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode.define_schema()
    assert [output.id for output in mel_schema.outputs] == ["vocals", "instruments"]


def test_audio_poc_request_mapping_and_named_result_decoding(monkeypatch):
    sync = AsyncMock(return_value=ComfyCloudGenerateResponse(task_id="task-audio", status="queued", polling_url="/tasks/task-audio", cancel_url="/tasks/task-audio/cancel"))
    poll = AsyncMock(return_value=ComfyCloudStatusResponse(task_id="task-audio", status="completed", output_urls={"vocals": "/vocals.mp3", "instruments": "/instruments.mp3"}))
    upload = AsyncMock(return_value="/uploads/song.m4a")
    download = AsyncMock(side_effect=["vocals-audio", "instruments-audio"])
    monkeypatch.setattr(nodes_comfy_cloud, "sync_op", sync)
    monkeypatch.setattr(nodes_comfy_cloud, "poll_op", poll)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    monkeypatch.setattr(nodes_comfy_cloud, "download_url_to_audio_input", download)
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}

    output = asyncio.run(nodes_comfy_cloud.ComfyCloudMelBandRoFormerStemSeparationNode.execute(audio))

    request = sync.call_args.kwargs["data"]
    assert request.workflow == "audio.melbandroformer-stem-separation.v1"
    assert request.inputs.model_dump(exclude_none=True) == {"assets": {"audio": {"type": "AUDIO", "url": "/uploads/song.m4a"}}}
    assert [call.args[0] for call in download.await_args_list] == ["/vocals.mp3", "/instruments.mp3"]
    assert poll.call_args.kwargs["cancel_endpoint"].path == "/tasks/task-audio/cancel"
    assert tuple(output) == ("vocals-audio", "instruments-audio")


def test_chatterbox_audio_inputs_use_named_staged_assets(monkeypatch):
    run = AsyncMock(return_value=("audio-output",))
    upload = AsyncMock(side_effect=["/uploads/source.m4a", "/uploads/target.m4a"])
    monkeypatch.setattr(nodes_comfy_cloud, "_run_audio_workflow", run)
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    source = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}
    target = {"waveform": torch.zeros(1, 1, 96000), "sample_rate": 48000}

    asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxVoiceConversionNode.execute(source, target, 7))

    inputs = run.call_args.args[2]
    assert inputs.model_dump(exclude_none=True) == {
        "assets": {
            "source_audio": {"type": "AUDIO", "url": "/uploads/source.m4a"},
            "target_voice_reference": {"type": "AUDIO", "url": "/uploads/target.m4a"},
        },
        "seed": 7,
    }
    assert "audio_url" not in inputs.model_dump(exclude_none=True)


def test_chatterbox_dialogue_rejects_invalid_speaker_labels(monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(nodes_comfy_cloud, "upload_audio_to_comfyapi", upload)
    audio = {"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000}

    with pytest.raises(ValueError, match="Every nonblank utterance"):
        asyncio.run(nodes_comfy_cloud.ComfyCloudChatterboxDialogueNode.execute("NARRATOR: Hello", audio, audio, 0.5, 0.5, 0.8, 0))
    upload.assert_not_awaited()


@pytest.mark.parametrize(("file_format", "expected_format"), [(".GLB", "glb"), ("SPZ", "spz")])
def test_download_cloud_3d_url_to_file_3d(monkeypatch, file_format, expected_format):
    node = nodes_comfy_cloud.ComfyCloudTextToImageNode
    downloaded = b"3d result"
    calls = []

    async def download(url, dest, **kwargs):
        calls.append((url, dest, kwargs))
        dest.write(downloaded)
        dest.seek(0)

    monkeypatch.setattr(download_helpers, "download_url_to_bytesio", download)

    output = asyncio.run(
        download_helpers.download_url_to_file_3d(
            f"/proxy/comfy-cloud/results/task-1/model.{expected_format}",
            file_format,
            timeout=45,
            max_retries=3,
            cls=node,
        )
    )

    assert output.format == expected_format
    assert output.get_bytes() == downloaded
    assert calls[0][0] == f"/proxy/comfy-cloud/results/task-1/model.{expected_format}"
    assert isinstance(calls[0][1], BytesIO)
    assert calls[0][2] == {"timeout": 45, "max_retries": 3, "cls": node}
