import asyncio
from io import BytesIO
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
