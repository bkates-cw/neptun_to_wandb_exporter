# #
# # Copyright (c) 2025, Neptune Labs Sp. z o.o.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import os
# import re
# import logging
# import tempfile
# from decimal import Decimal
# from pathlib import Path
# from typing import Generator, Optional, Any
# import pandas as pd
# import pyarrow as pa
# import wandb

# from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
# from neptune_exporter.loaders.loader import DataLoader


# class WandBLoader(DataLoader):
#     """Loads Neptune data from parquet files into Weights & Biases."""

#     def __init__(
#         self,
#         entity: str,
#         api_key: Optional[str] = None,
#         name_prefix: Optional[str] = None,
#         show_client_logs: bool = False,
#     ):
#         """
#         Initialize W&B loader.

#         Args:
#             entity: W&B entity (organization/username)
#             api_key: Optional W&B API key for authentication
#             name_prefix: Optional prefix for project and run names
#             verbose: Enable verbose logging
#         """
#         self.entity = entity
#         self.name_prefix = name_prefix
#         self._logger = logging.getLogger(__name__)
#         self._active_run: Optional[wandb.Run] = None

#         # Authenticate with W&B
#         if api_key:
#             wandb.login(key=api_key)

#         # Configure W&B logging
#         if not show_client_logs:
#             os.environ["WANDB_SILENT"] = "true"

#     def _sanitize_attribute_name(self, attribute_path: str) -> str:
#         """
#         Sanitize Neptune attribute path to W&B-compatible key.

#         W&B key constraints:
#         - Must start with a letter or underscore
#         - Can only contain letters, numbers, and underscores
#         - Pattern: /^[_a-zA-Z][_a-zA-Z0-9]*$/
#         """
#         # Replace invalid characters with underscores
#         sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", attribute_path)

#         # Ensure it starts with a letter or underscore
#         if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
#             sanitized = "_" + sanitized

#         # Handle empty result
#         if not sanitized:
#             sanitized = "_attribute"

#         return sanitized

#     def _get_project_name(self, project_id: str) -> str:
#         """Get W&B project name from Neptune project ID."""
#         # W&B uses entity/project structure
#         # Neptune project_id maps directly to W&B project
#         name = project_id

#         if self.name_prefix:
#             name = f"{self.name_prefix}_{name}"

#         # Sanitize for W&B project name (alphanumeric, hyphens, underscores)
#         name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

#         return name

#     def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
#         """Convert Neptune decimal step to W&B integer step."""
#         if step is None:
#             return 0
#         return int(float(step) * step_multiplier)

#     def create_experiment(
#         self, project_id: str, experiment_name: str
#     ) -> TargetExperimentId:
#         """
#         Neptune experiment_name maps to W&B group (set in create_run).
#         We return the experiment name as the group name to use.
#         """
#         return TargetExperimentId(experiment_name)

#     def find_run(
#         self,
#         project_id: ProjectId,
#         run_name: str,
#         experiment_id: Optional[TargetExperimentId],
#     ) -> Optional[TargetRunId]:
#         """Find a run by name in a W&B project.

#         Args:
#             run_name: Name of the run to find
#             experiment_id: W&B group name (experiment name from Neptune)
#             project_id: Neptune project ID (used to construct W&B project name)

#         Returns:
#             W&B run ID if found, None otherwise
#         """
#         sanitized_project = self._get_project_name(project_id)

#         try:
#             # Use wandb.Api() to search for runs
#             api = wandb.Api()
#             project_path = f"{self.entity}/{sanitized_project}"

#             # Search for runs with matching name and group
#             filters = {"display_name": run_name}
#             if experiment_id:
#                 filters["group"] = experiment_id

#             runs = api.runs(project_path, filters=filters, per_page=1)

#             # Get the first matching run
#             for run in runs:
#                 return TargetRunId(run.id)

#             return None
#         except Exception:
#             self._logger.error(
#                 f"Error finding project {project_id}, run '{run_name}'",
#                 exc_info=True,
#             )
#             return None

#     def create_run(
#         self,
#         project_id: ProjectId,
#         run_name: str,
#         experiment_id: Optional[TargetExperimentId] = None,
#         parent_run_id: Optional[TargetRunId] = None,
#         fork_step: Optional[float] = None,
#         step_multiplier: Optional[int] = None,
#     ) -> TargetRunId:
#         """Create W&B run, with support for forked runs.

#         Args:
#             fork_step: Fork step as float (decimal). Will be converted to int using step_multiplier.
#             step_multiplier: Step multiplier for converting decimal steps to integers.
#                 If provided, will be used for fork_step conversion. If not provided,
#                 will calculate from fork_step alone as fallback.
#         """
#         sanitized_project = self._get_project_name(project_id)

#         try:
#             # Prepare init arguments
#             init_kwargs: dict[str, Any] = {
#                 "entity": self.entity,
#                 "project": sanitized_project,
#                 "group": experiment_id,
#                 "name": run_name,
#             }

#             # Handle forking if parent exists
#             if parent_run_id:
#                 # Convert fork_step to int using provided step_multiplier
#                 # step_multiplier should always be provided when fork_step is set
#                 if fork_step is not None:
#                     if step_multiplier is None:
#                         raise ValueError(
#                             "step_multiplier must be provided when fork_step is set"
#                         )
#                     step_int = self._convert_step_to_int(
#                         Decimal(str(fork_step)), step_multiplier
#                     )
#                 else:
#                     step_int = 0

#                 # W&B fork format: entity/project/run_id?_step=step
#                 fork_from = f"{self.entity}/{sanitized_project}/{parent_run_id}?_step={step_int}"
#                 init_kwargs["fork_from"] = fork_from
#                 self._logger.info(
#                     f"Creating forked run '{run_name}' from parent {parent_run_id} at step {step_int}"
#                 )

#             # Initialize the run
#             run = wandb.init(**init_kwargs)
#             wandb_run_id = run.id

#             self._active_run = run

#             self._logger.info(f"Created run '{run_name}' with W&B ID {wandb_run_id}")
#             return TargetRunId(wandb_run_id)

#         except Exception:
#             self._logger.error(
#                 f"Error creating project {project_id}, run '{run_name}'",
#                 exc_info=True,
#             )
#             raise

#     def upload_run_data(
#         self,
#         run_data: Generator[pa.Table, None, None],
#         run_id: TargetRunId,
#         files_directory: Path,
#         step_multiplier: int,
#     ) -> None:
#         """Upload all data for a single run to W&B.

#         Args:
#             step_multiplier: Step multiplier for converting decimal steps to integers
#         """
#         try:
#             # Note: We assume the run is already active from create_run
#             # If not, we would need to resume it
#             if self._active_run is None or self._active_run.id != run_id:
#                 self._logger.error(
#                     f"Run {run_id} is not active. Call create_run first."
#                 )
#                 raise RuntimeError(f"Run {run_id} is not active")

#             for run_data_part in run_data:
#                 run_df = run_data_part.to_pandas()

#                 self.upload_parameters(run_df, run_id)
#                 self.upload_metrics(run_df, run_id, step_multiplier)
#                 self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

#             # Finish the run
#             self._active_run.finish()
#             self._active_run = None

#             self._logger.info(f"Successfully uploaded run {run_id} to W&B")

#         except Exception:
#             self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
#             if self._active_run:
#                 self._active_run.finish(exit_code=1)
#                 self._active_run = None
#             raise

#     def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
#         """Upload parameters (configs) to W&B run."""
#         if self._active_run is None:
#             raise RuntimeError("No active run")

#         param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
#         param_data = run_data[run_data["attribute_type"].isin(param_types)]

#         if param_data.empty:
#             return

#         config = {}
#         for _, row in param_data.iterrows():
#             attr_name = self._sanitize_attribute_name(row["attribute_path"])

#             # Get the appropriate value based on attribute type
#             if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
#                 config[attr_name] = row["float_value"]
#             elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
#                 config[attr_name] = int(row["int_value"])
#             elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
#                 config[attr_name] = row["string_value"]
#             elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
#                 config[attr_name] = bool(row["bool_value"])
#             elif row["attribute_type"] == "datetime" and pd.notna(
#                 row["datetime_value"]
#             ):
#                 config[attr_name] = str(row["datetime_value"])
#             elif (
#                 row["attribute_type"] == "string_set"
#                 and row["string_set_value"] is not None
#             ):
#                 config[attr_name] = list(row["string_set_value"])

#         if config:
#             self._active_run.config.update(config)
#             self._logger.info(f"Uploaded {len(config)} parameters for run {run_id}")

#     def upload_metrics(
#         self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
#     ) -> None:
#         """Upload metrics (float series) to W&B run.

#         Args:
#             step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
#         """
#         if self._active_run is None:
#             raise RuntimeError("No active run")

#         metrics_data = run_data[run_data["attribute_type"] == "float_series"]

#         if metrics_data.empty:
#             return

#         # Use global step multiplier (calculated from all series + fork_step)
#         # Group by step to log all metrics at each step together
#         for step_value, group in metrics_data.groupby("step"):
#             if pd.notna(step_value):
#                 step = self._convert_step_to_int(step_value, step_multiplier)

#                 metrics = {}
#                 for _, row in group.iterrows():
#                     if pd.notna(row["float_value"]):
#                         attr_name = self._sanitize_attribute_name(row["attribute_path"])
#                         metrics[attr_name] = row["float_value"]

#                 if metrics:
#                     self._active_run.log(metrics, step=step)

#         self._logger.info(f"Uploaded metrics for run {run_id}")

#     def upload_artifacts(
#         self,
#         run_data: pd.DataFrame,
#         run_id: TargetRunId,
#         files_base_path: Path,
#         step_multiplier: int,
#     ) -> None:
#         """Upload files and series as artifacts to W&B run.

#         Args:
#             step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
#         """
#         if self._active_run is None:
#             raise RuntimeError("No active run")

#         # Handle regular files
#         file_data = run_data[
#             run_data["attribute_type"].isin(["file", "file_set", "artifact"])
#         ]
#         for _, row in file_data.iterrows():
#             if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
#                 file_path = files_base_path / row["file_value"]["path"]
#                 if file_path.exists():
#                     attr_name = self._sanitize_attribute_name(row["attribute_path"])
#                     artifact = wandb.Artifact(
#                         name=attr_name, type=row["attribute_type"]
#                     )
#                     if file_path.is_file():
#                         artifact.add_file(str(file_path))
#                     else:
#                         artifact.add_dir(str(file_path))
#                     self._active_run.log_artifact(artifact)
#                 else:
#                     self._logger.warning(f"File not found: {file_path}")

#         # Handle file series
#         file_series_data = run_data[run_data["attribute_type"] == "file_series"]
#         for attr_path, group in file_series_data.groupby("attribute_path"):
#             attr_name = self._sanitize_attribute_name(attr_path)

#             for _, row in group.iterrows():
#                 if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
#                     file_path = files_base_path / row["file_value"]["path"]
#                     if file_path.exists():
#                         step = (
#                             self._convert_step_to_int(row["step"], step_multiplier)
#                             if pd.notna(row["step"])
#                             else 0
#                         )
#                         artifact_name = f"{attr_name}_step_{step}"
#                         artifact = wandb.Artifact(
#                             name=artifact_name, type="file_series"
#                         )
#                         if file_path.is_file():
#                             artifact.add_file(str(file_path))
#                         else:
#                             artifact.add_dir(str(file_path))
#                         self._active_run.log_artifact(artifact)
#                     else:
#                         self._logger.warning(f"File not found: {file_path}")

#         # Handle string series as text artifacts
#         string_series_data = run_data[run_data["attribute_type"] == "string_series"]
#         for attr_path, group in string_series_data.groupby("attribute_path"):
#             attr_name = self._sanitize_attribute_name(attr_path)

#             # Create temporary file with text content
#             with tempfile.NamedTemporaryFile(
#                 mode="w", suffix=".txt", encoding="utf-8"
#             ) as tmp_file:
#                 for _, row in group.iterrows():
#                     if pd.notna(row["string_value"]):
#                         series_step = (
#                             self._convert_step_to_int(row["step"], step_multiplier)
#                             if pd.notna(row["step"])
#                             else None
#                         )
#                         timestamp = (
#                             row["timestamp"].isoformat()
#                             if pd.notna(row["timestamp"])
#                             else None
#                         )
#                         text_line = (
#                             f"{series_step}; {timestamp}; {row['string_value']}\n"
#                         )
#                         tmp_file.write(text_line)
#                 tmp_file_path = tmp_file.name

#                 # Create and log W&B artifact
#                 artifact = wandb.Artifact(name=attr_name, type="string_series")
#                 artifact.add_file(tmp_file_path, name="series.txt")
#                 self._active_run.log_artifact(artifact)

#         # Handle histogram series as W&B Histograms
#         histogram_series_data = run_data[
#             run_data["attribute_type"] == "histogram_series"
#         ]
#         for attr_path, group in histogram_series_data.groupby("attribute_path"):
#             attr_name = self._sanitize_attribute_name(attr_path)
#             # Use global step multiplier

#             for _, row in group.iterrows():
#                 if pd.notna(row["histogram_value"]) and isinstance(
#                     row["histogram_value"], dict
#                 ):
#                     step = (
#                         self._convert_step_to_int(row["step"], step_multiplier)
#                         if pd.notna(row["step"])
#                         else 0
#                     )
#                     hist = row["histogram_value"]

#                     # Convert Neptune histogram to W&B Histogram
#                     # Neptune format: {"type": str, "edges": list, "values": list}
#                     # W&B expects histogram data as np_histogram tuple or sequence
#                     try:
#                         wandb_hist = wandb.Histogram(
#                             np_histogram=(hist.get("values", []), hist.get("edges", []))
#                         )
#                         self._active_run.log({attr_name: wandb_hist}, step=step)
#                     except Exception:
#                         self._logger.error(
#                             f"Failed to log histogram for {attr_path} at step {step}",
#                             exc_info=True,
#                         )

#         self._logger.info(f"Uploaded artifacts for run {run_id}")


#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional, Any
import pandas as pd
import pyarrow as pa
import wandb
import magic
import ffmpeg

from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
from neptune_exporter.loaders.loader import DataLoader


# Rich media file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tiff", ".tif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TABLE_EXTENSIONS = {".csv", ".tsv"}
HTML_EXTENSIONS = {".html", ".htm"}


class WandBLoader(DataLoader):
    """Loads Neptune data from parquet files into Weights & Biases."""

    def __init__(
        self,
        entity: str,
        api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
        rich: bool = False,
    ):
        """
        Initialize W&B loader.

        Args:
            entity: W&B entity (organization/username)
            api_key: Optional W&B API key for authentication
            name_prefix: Optional prefix for project and run names
            show_client_logs: Enable verbose logging
            rich: Upload files as native W&B media types (images, audio, video, tables)
        """
        self.entity = entity
        self.name_prefix = name_prefix
        self.rich = rich
        self._logger = logging.getLogger(__name__)
        self._active_run: Optional[wandb.Run] = None

        # Authenticate with W&B
        if api_key:
            wandb.login(key=api_key)

        # Configure W&B logging
        if not show_client_logs:
            os.environ["WANDB_SILENT"] = "true"

        if self.rich:
            self._logger.info("Rich mode enabled - uploading as native W&B media types")

    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type using extension, magic bytes, and directory hints."""
        ext = file_path.suffix.lower()
        print(f"[DEBUG _get_file_type] file_path: {file_path}, ext: '{ext}'")

        # Step 1: Try extension first (fastest)
        if ext in IMAGE_EXTENSIONS:
            print(f"[DEBUG _get_file_type] Matched IMAGE by extension")
            return "image"
        elif ext in AUDIO_EXTENSIONS:
            print(f"[DEBUG _get_file_type] Matched AUDIO by extension")
            return "audio"
        elif ext in VIDEO_EXTENSIONS:
            print(f"[DEBUG _get_file_type] Matched VIDEO by extension")
            return "video"
        elif ext in TABLE_EXTENSIONS:
            print(f"[DEBUG _get_file_type] Matched TABLE by extension")
            return "table"
        elif ext in HTML_EXTENSIONS:
            print(f"[DEBUG _get_file_type] Matched HTML by extension")
            return "html"

        # Step 2: No extension or unknown extension - use magic bytes
        print(f"[DEBUG _get_file_type] No extension match, trying magic bytes...")
        try:
            mime = magic.from_file(str(file_path), mime=True)
            print(f"[DEBUG _get_file_type] Detected MIME type: {mime}")

            # Map MIME types to our categories
            if mime.startswith('image/'):
                # Special case: GIFs might be in video/ folder but are images for W&B
                parent_dir = file_path.parent.name
                if mime == 'image/gif' and parent_dir == 'video':
                    print(f"[DEBUG _get_file_type] GIF in video folder - treating as video")
                    return "video"
                print(f"[DEBUG _get_file_type] Matched IMAGE by magic")
                return "image"
            elif mime.startswith('audio/'):
                print(f"[DEBUG _get_file_type] Matched AUDIO by magic")
                return "audio"
            elif mime.startswith('video/'):
                print(f"[DEBUG _get_file_type] Matched VIDEO by magic")
                return "video"
            elif mime in ('text/csv', 'text/tab-separated-values'):
                print(f"[DEBUG _get_file_type] Matched TABLE by magic")
                return "table"
            elif mime == 'text/html':
                print(f"[DEBUG _get_file_type] Matched HTML by magic")
                return "html"
        except Exception as e:
            print(f"[DEBUG _get_file_type] Magic bytes detection failed: {e}")

        # Step 3: Use directory name as hint (common sense!)
        parent_dir = file_path.parent.name
        print(f"[DEBUG _get_file_type] Trying directory hint: {parent_dir}")

        if parent_dir in ('images', 'image', 'visualizations'):
            print(f"[DEBUG _get_file_type] Directory suggests IMAGE")
            return "image"
        elif parent_dir == 'audio':
            print(f"[DEBUG _get_file_type] Directory suggests AUDIO")
            return "audio"
        elif parent_dir == 'video':
            print(f"[DEBUG _get_file_type] Directory suggests VIDEO")
            return "video"
        elif parent_dir in ('tables', 'table'):
            print(f"[DEBUG _get_file_type] Directory suggests TABLE")
            return "table"

        print(f"[DEBUG _get_file_type] No match - returning 'file'")
        return "file"

    def _reencode_video_to_h264(self, input_path: Path) -> Path:
        """Re-encode video to H.264 MP4 for browser compatibility.

        Returns path to the re-encoded video (a temp file).
        """
        print(f"[DEBUG _reencode_video_to_h264] Re-encoding video: {input_path}")

        # Create temp file for output
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        output_path = temp_file.name

        try:
            # Re-encode to H.264 with web-compatible settings
            (
                ffmpeg
                .input(str(input_path))
                .output(
                    output_path,
                    vcodec='libx264',  # H.264 codec
                    acodec='aac',      # AAC audio codec
                    **{
                        'preset': 'fast',           # Fast encoding
                        'crf': '23',                # Quality (lower = better, 23 is good)
                        'movflags': '+faststart',   # Enable web streaming
                        'pix_fmt': 'yuv420p',       # Compatible pixel format
                    }
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            print(f"[DEBUG _reencode_video_to_h264] Successfully re-encoded to: {output_path}")
            return Path(output_path)
        except ffmpeg.Error as e:
            print(f"[DEBUG _reencode_video_to_h264] FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            # Clean up on failure
            if Path(output_path).exists():
                os.unlink(output_path)
            raise

    def _get_extension_for_mime(self, mime: str) -> str:
        """Map MIME type to file extension for W&B."""
        mime_to_ext = {
            # Images
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/webp': '.webp',
            'image/svg+xml': '.svg',
            'image/tiff': '.tiff',
            # Audio
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/wave': '.wav',
            'audio/x-wav': '.wav',
            'audio/ogg': '.ogg',
            'audio/flac': '.flac',
            'audio/aac': '.aac',
            'audio/mp4': '.m4a',
            # Video
            'video/mp4': '.mp4',
            'video/mpeg': '.mpeg',
            'video/quicktime': '.mov',
            'video/x-msvideo': '.avi',
            'video/webm': '.webm',
            'video/x-matroska': '.mkv',
        }
        return mime_to_ext.get(mime, '')

    def _upload_rich_file(self, file_path: Path, attr_name: str, step: Optional[int] = None) -> bool:
        """Upload file as native W&B type. Returns True if successful."""
        print(f"\n[DEBUG _upload_rich_file] Called with:")
        print(f"  - file_path: {file_path}")
        print(f"  - attr_name: {attr_name}")
        print(f"  - step: {step}")
        print(f"  - file exists: {file_path.exists()}")
        print(f"  - is file: {file_path.is_file() if file_path.exists() else 'N/A'}")

        if not file_path.exists() or not file_path.is_file():
            print(f"[DEBUG _upload_rich_file] File doesn't exist or isn't a file - returning False")
            return False

        file_type = self._get_file_type(file_path)
        print(f"[DEBUG _upload_rich_file] Detected file_type: {file_type}")

        # Check if file already has an extension
        has_extension = bool(file_path.suffix)
        temp_file = None
        reencoded_video = None

        try:
            # For audio/video without extensions, W&B needs the extension in the filename
            # Create a temp file with proper extension
            if not has_extension and file_type in ('audio', 'video'):
                print(f"[DEBUG _upload_rich_file] File has no extension, detecting MIME type...")
                mime = magic.from_file(str(file_path), mime=True)
                ext = self._get_extension_for_mime(mime)
                print(f"[DEBUG _upload_rich_file] MIME: {mime}, Extension: {ext}")

                if ext:
                    # Create temp file with proper extension
                    import shutil
                    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                    temp_file.close()
                    shutil.copy2(file_path, temp_file.name)
                    file_path = Path(temp_file.name)
                    print(f"[DEBUG _upload_rich_file] Created temp file with extension: {file_path}")
                else:
                    print(f"[DEBUG _upload_rich_file] Could not determine extension for MIME: {mime}")
                    return False

            if file_type == "image":
                print(f"[DEBUG _upload_rich_file] Creating wandb.Image for {file_path}")
                media = wandb.Image(str(file_path))
            elif file_type == "audio":
                print(f"[DEBUG _upload_rich_file] Creating wandb.Audio for {file_path}")
                media = wandb.Audio(str(file_path))
            elif file_type == "video":
                # Re-encode video to H.264 for browser compatibility
                print(f"[DEBUG _upload_rich_file] Re-encoding video to H.264 for browser compatibility...")
                reencoded_video = self._reencode_video_to_h264(file_path)
                print(f"[DEBUG _upload_rich_file] Creating wandb.Video for {reencoded_video}")
                media = wandb.Video(str(reencoded_video))
            elif file_type == "table":
                print(f"[DEBUG _upload_rich_file] Creating wandb.Table for {file_path}")
                df = pd.read_csv(file_path) if file_path.suffix == ".csv" else pd.read_csv(file_path, sep="\t")
                media = wandb.Table(dataframe=df)
            elif file_type == "html":
                # Check if this is an HTML table that should be converted to wandb.Table
                parent_dir = file_path.parent.name
                if parent_dir in ('tables', 'table'):
                    print(f"[DEBUG _upload_rich_file] HTML in tables dir - parsing to wandb.Table: {file_path}")
                    try:
                        # Parse HTML table back to DataFrame
                        dfs = pd.read_html(str(file_path))
                        if dfs:
                            # Force DataFrame to materialize by making a copy
                            df = dfs[0].copy()
                            print(f"[DEBUG _upload_rich_file] Parsed HTML table with {len(df)} rows, {len(df.columns)} columns")
                            print(f"[DEBUG _upload_rich_file] Columns: {list(df.columns)}")
                            media = wandb.Table(dataframe=df)
                            print(f"[DEBUG _upload_rich_file] Created wandb.Table successfully")
                        else:
                            print(f"[DEBUG _upload_rich_file] No tables found in HTML, using wandb.Html")
                            with open(file_path, "r", encoding="utf-8") as f:
                                media = wandb.Html(f.read())
                    except Exception as e:
                        print(f"[DEBUG _upload_rich_file] Failed to parse HTML table: {e}, using wandb.Html")
                        with open(file_path, "r", encoding="utf-8") as f:
                            media = wandb.Html(f.read())
                else:
                    print(f"[DEBUG _upload_rich_file] Creating wandb.Html for {file_path}")
                    with open(file_path, "r", encoding="utf-8") as f:
                        media = wandb.Html(f.read())
            else:
                print(f"[DEBUG _upload_rich_file] Unknown file_type '{file_type}' - returning False")
                return False

            print(f"[DEBUG _upload_rich_file] Logging media to wandb with attr_name: {attr_name}")
            if step is not None:
                self._active_run.log({attr_name: media, "step": step})
            else:
                self._active_run.log({attr_name: media})
            print(f"[DEBUG _upload_rich_file] Successfully logged! Returning True")
            return True

        except Exception as e:
            print(f"[DEBUG _upload_rich_file] EXCEPTION caught: {e}")
            print(f"[DEBUG _upload_rich_file] Exception type: {type(e)}")
            import traceback
            print(f"[DEBUG _upload_rich_file] Traceback:\n{traceback.format_exc()}")
            self._logger.warning(f"Failed to upload rich file {file_path}: {e}")
            return False
        finally:
            # Clean up temp files if created
            if temp_file and Path(temp_file.name).exists():
                print(f"[DEBUG _upload_rich_file] Cleaning up temp file: {temp_file.name}")
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    print(f"[DEBUG _upload_rich_file] Failed to delete temp file: {e}")

            if reencoded_video and reencoded_video.exists():
                print(f"[DEBUG _upload_rich_file] Cleaning up re-encoded video: {reencoded_video}")
                try:
                    os.unlink(reencoded_video)
                except Exception as e:
                    print(f"[DEBUG _upload_rich_file] Failed to delete re-encoded video: {e}")

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """
        Sanitize Neptune attribute path to W&B-compatible key.

        W&B key constraints:
        - Must start with a letter or underscore
        - Can only contain letters, numbers, and underscores
        - Pattern: /^[_a-zA-Z][_a-zA-Z0-9]*$/
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", attribute_path)

        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Handle empty result
        if not sanitized:
            sanitized = "_attribute"

        return sanitized

    def _get_project_name(self, project_id: str) -> str:
        """Get W&B project name from Neptune project ID."""
        # W&B uses entity/project structure
        # Neptune project_id maps directly to W&B project
        name = project_id

        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize for W&B project name (alphanumeric, hyphens, underscores)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to W&B integer step."""
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def create_experiment(
        self, project_id: str, experiment_name: str
    ) -> TargetExperimentId:
        """
        Neptune experiment_name maps to W&B group (set in create_run).
        We return the experiment name as the group name to use.
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find a run by name in a W&B project.

        Args:
            run_name: Name of the run to find
            experiment_id: W&B group name (experiment name from Neptune)
            project_id: Neptune project ID (used to construct W&B project name)

        Returns:
            W&B run ID if found, None otherwise
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Use wandb.Api() to search for runs
            api = wandb.Api()
            project_path = f"{self.entity}/{sanitized_project}"

            # Search for runs with matching name and group
            filters = {"display_name": run_name}
            if experiment_id:
                filters["group"] = experiment_id

            runs = api.runs(project_path, filters=filters, per_page=1)

            # Get the first matching run
            for run in runs:
                return TargetRunId(run.id)

            return None
        except Exception:
            self._logger.error(
                f"Error finding project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """Create W&B run, with support for forked runs.

        Args:
            fork_step: Fork step as float (decimal). Will be converted to int using step_multiplier.
            step_multiplier: Step multiplier for converting decimal steps to integers.
                If provided, will be used for fork_step conversion. If not provided,
                will calculate from fork_step alone as fallback.
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Prepare init arguments
            init_kwargs: dict[str, Any] = {
                "entity": self.entity,
                "project": sanitized_project,
                "group": experiment_id,
                "name": run_name,
            }

            # Handle forking if parent exists
            if parent_run_id:
                # Convert fork_step to int using provided step_multiplier
                # step_multiplier should always be provided when fork_step is set
                if fork_step is not None:
                    if step_multiplier is None:
                        raise ValueError(
                            "step_multiplier must be provided when fork_step is set"
                        )
                    step_int = self._convert_step_to_int(
                        Decimal(str(fork_step)), step_multiplier
                    )
                else:
                    step_int = 0

                # W&B fork format: entity/project/run_id?_step=step
                fork_from = f"{self.entity}/{sanitized_project}/{parent_run_id}?_step={step_int}"
                init_kwargs["fork_from"] = fork_from
                self._logger.info(
                    f"Creating forked run '{run_name}' from parent {parent_run_id} at step {step_int}"
                )

            # Initialize the run
            run = wandb.init(**init_kwargs)
            # run.define_metric("*", step_sync=False)

            wandb_run_id = run.id

            self._active_run = run

            self._logger.info(f"Created run '{run_name}' with W&B ID {wandb_run_id}")
            return TargetRunId(wandb_run_id)

        except Exception:
            self._logger.error(
                f"Error creating project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """Upload all data for a single run to W&B.

        Args:
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        try:
            # Note: We assume the run is already active from create_run
            # If not, we would need to resume it
            if self._active_run is None or self._active_run.id != run_id:
                self._logger.error(
                    f"Run {run_id} is not active. Call create_run first."
                )
                raise RuntimeError(f"Run {run_id} is not active")

            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()

                self.upload_parameters(run_df, run_id)
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

            # Finish the run
            self._active_run.finish()
            self._active_run = None

            self._logger.info(f"Successfully uploaded run {run_id} to W&B")

        except Exception:
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            if self._active_run:
                self._active_run.finish(exit_code=1)
                self._active_run = None
            raise

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters (configs) to W&B run."""
        if self._active_run is None:
            raise RuntimeError("No active run")

        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        config = {}
        for _, row in param_data.iterrows():
            attr_name = self._sanitize_attribute_name(row["attribute_path"])

            # Get the appropriate value based on attribute type
            if row["attribute_type"] == "float" and pd.notna(row["float_value"]):
                config[attr_name] = row["float_value"]
            elif row["attribute_type"] == "int" and pd.notna(row["int_value"]):
                config[attr_name] = int(row["int_value"])
            elif row["attribute_type"] == "string" and pd.notna(row["string_value"]):
                config[attr_name] = row["string_value"]
            elif row["attribute_type"] == "bool" and pd.notna(row["bool_value"]):
                config[attr_name] = bool(row["bool_value"])
            elif row["attribute_type"] == "datetime" and pd.notna(
                row["datetime_value"]
            ):
                config[attr_name] = str(row["datetime_value"])
            elif (
                row["attribute_type"] == "string_set"
                and row["string_set_value"] is not None
            ):
                config[attr_name] = list(row["string_set_value"])

        if config:
            self._active_run.config.update(config)
            self._logger.info(f"Uploaded {len(config)} parameters for run {run_id}")

    def upload_metrics(
        self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int
    ) -> None:
        """Upload metrics (float series) to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Use global step multiplier (calculated from all series + fork_step)
        # Group by step to log all metrics at each step together
        for step_value, group in metrics_data.groupby("step"):
            if pd.notna(step_value):
                step = self._convert_step_to_int(step_value, step_multiplier)

                metrics = {}
                for _, row in group.iterrows():
                    if pd.notna(row["float_value"]):
                        attr_name = self._sanitize_attribute_name(row["attribute_path"])
                        metrics[attr_name] = row["float_value"]

                if metrics:
                    self._active_run.log(metrics, step=step)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as artifacts to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        # Handle regular files
        file_data = run_data[
            run_data["attribute_type"].isin(["file", "file_set", "artifact"])
        ]
        print(f"\n[DEBUG upload_artifacts] Processing {len(file_data)} regular files")
        print(f"[DEBUG upload_artifacts] self.rich = {self.rich}")

        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                print(f"\n[DEBUG upload_artifacts] Processing file: {file_path}")
                print(f"[DEBUG upload_artifacts] attribute_path: {row['attribute_path']}")
                print(f"[DEBUG upload_artifacts] attribute_type: {row['attribute_type']}")
                if file_path.exists():
                    attr_name = self._sanitize_attribute_name(row["attribute_path"])

                    # Try rich upload first if enabled
                    if self.rich and file_path.is_file():
                        print(f"[DEBUG upload_artifacts] Attempting rich upload...")
                        if self._upload_rich_file(file_path, attr_name):
                            print(f"[DEBUG upload_artifacts] Rich upload succeeded, skipping artifact")
                            continue
                        print(f"[DEBUG upload_artifacts] Rich upload failed, falling back to artifact")
                    else:
                        print(f"[DEBUG upload_artifacts] Skipping rich upload (rich={self.rich}, is_file={file_path.is_file()})")

                    # Fall back to artifact
                    artifact = wandb.Artifact(
                        name=attr_name, type=row["attribute_type"]
                    )
                    if file_path.is_file():
                        artifact.add_file(str(file_path))
                    else:
                        artifact.add_dir(str(file_path))
                    self._active_run.log_artifact(artifact)
                else:
                    self._logger.warning(f"File not found: {file_path}")

        # Handle file series
        file_series_data = run_data[run_data["attribute_type"] == "file_series"]
        print(f"\n[DEBUG upload_artifacts] Processing {len(file_series_data)} file_series items")

        for attr_path, group in file_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)
            print(f"\n[DEBUG upload_artifacts] Processing file_series group: {attr_path}")
            print(f"[DEBUG upload_artifacts] Group has {len(group)} items")

            # Try rich mode: collect all files and create a table
            if self.rich:
                print(f"[DEBUG upload_artifacts] Rich mode: collecting files for table...")
                table_rows = []
                all_files_exist = True
                file_type = None

                for _, row in group.iterrows():
                    if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                        file_path = files_base_path / row["file_value"]["path"]
                        if file_path.exists() and file_path.is_file():
                            step = (
                                self._convert_step_to_int(row["step"], step_multiplier)
                                if pd.notna(row["step"])
                                else 0
                            )

                            # Detect file type from first file
                            if file_type is None:
                                file_type = self._get_file_type(file_path)
                                print(f"[DEBUG upload_artifacts] Detected file_series type: {file_type}")

                            table_rows.append({
                                "step": step,
                                "file_path": file_path,
                                "row": row
                            })
                        else:
                            all_files_exist = False
                            self._logger.warning(f"File not found or not a file: {file_path}")

                # If we have files and they're media (image/video/audio), create a table
                if table_rows and file_type in ("image", "video", "audio") and all_files_exist:
                    print(f"[DEBUG upload_artifacts] Creating wandb.Table for {len(table_rows)} {file_type}s")
                    try:
                        # Sort by step
                        table_rows.sort(key=lambda x: x["step"])

                        # Create table with descriptive column name
                        if file_type == "image":
                            table = wandb.Table(columns=["step", "image"])
                        elif file_type == "video":
                            table = wandb.Table(columns=["step", "video"])
                        elif file_type == "audio":
                            table = wandb.Table(columns=["step", "audio"])
                        else:
                            table = wandb.Table(columns=["step", "file"])

                        temp_files_to_cleanup = []  # Collect temp files to delete after logging

                        for row_data in table_rows:
                            file_path = row_data["file_path"]
                            step = row_data["step"]

                            # Ensure step is a plain Python int
                            step_int = int(step)

                            # Create appropriate wandb media object and add to table
                            # Use EXACT same pattern as upload_media.py
                            if file_type == "image":
                                print(f"[DEBUG] Creating wandb.Image from: {file_path}")
                                from PIL import Image as PILImage
                                import numpy as np
                                img = PILImage.open(file_path)
                                # Convert to numpy array to force loading (PIL.open is lazy)
                                img_array = np.array(img)
                                table.add_data(step_int, wandb.Image(img_array, caption=f"Step {step_int}"))
                                print(f"[DEBUG] Added image at step {step_int} to table")
                            elif file_type == "video":
                                print(f"[DEBUG] Creating wandb.Video from: {file_path}")
                                # Re-encode video for browser compatibility
                                reencoded = self._reencode_video_to_h264(file_path)
                                table.add_data(step_int, wandb.Video(str(reencoded), caption=f"Step {step_int}"))
                                # Track temp file for cleanup AFTER table is logged
                                temp_files_to_cleanup.append(reencoded)
                                print(f"[DEBUG] Added video at step {step_int} to table")
                            elif file_type == "audio":
                                print(f"[DEBUG] Creating wandb.Audio from: {file_path}")
                                # Copy to temp file with proper extension if needed
                                if not file_path.suffix:
                                    import shutil
                                    mime = magic.from_file(str(file_path), mime=True)
                                    ext = self._get_extension_for_mime(mime)
                                    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                                    temp_file.close()
                                    shutil.copy2(file_path, temp_file.name)
                                    table.add_data(step_int, wandb.Audio(temp_file.name, caption=f"Step {step_int}"))
                                    # Track temp file for cleanup AFTER table is logged
                                    temp_files_to_cleanup.append(Path(temp_file.name))
                                else:
                                    table.add_data(step_int, wandb.Audio(str(file_path), caption=f"Step {step_int}"))
                                print(f"[DEBUG] Added audio at step {step_int} to table")

                        # Log the table FIRST (no step parameter - table IS the series)
                        self._active_run.log({attr_name: table})
                        print(f"[DEBUG upload_artifacts] Successfully logged table with {len(table_rows)} items")

                        # NOW clean up temp files
                        for temp_path in temp_files_to_cleanup:
                            if temp_path.exists():
                                print(f"[DEBUG upload_artifacts] Cleaning up temp file: {temp_path}")
                                os.unlink(temp_path)

                        continue  # Skip artifact upload for this group

                    except Exception as e:
                        print(f"[DEBUG upload_artifacts] Table creation failed: {e}")
                        import traceback
                        print(f"[DEBUG upload_artifacts] Traceback:\n{traceback.format_exc()}")
                        self._logger.warning(f"Failed to create table for {attr_path}: {e}")
                        # Fall through to artifact upload

            # Fall back to individual artifacts (non-rich mode or table creation failed)
            print(f"[DEBUG upload_artifacts] Falling back to artifact upload for file_series")
            for _, row in group.iterrows():
                if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                    file_path = files_base_path / row["file_value"]["path"]
                    if file_path.exists():
                        step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else 0
                        )

                        artifact_name = f"{attr_name}_step_{step}"
                        artifact = wandb.Artifact(
                            name=artifact_name, type="file_series"
                        )
                        if file_path.is_file():
                            artifact.add_file(str(file_path))
                        else:
                            artifact.add_dir(str(file_path))
                        self._active_run.log_artifact(artifact)
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle string series
        string_series_data = run_data[run_data["attribute_type"] == "string_series"]
        for attr_path, group in string_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            if self.rich:
                # Upload as W&B Table
                table_data = []
                for _, row in group.iterrows():
                    if pd.notna(row["string_value"]):
                        series_step = (
                            self._convert_step_to_int(row["step"], step_multiplier)
                            if pd.notna(row["step"])
                            else None
                        )
                        timestamp = (
                            row["timestamp"].isoformat()
                            if pd.notna(row["timestamp"])
                            else None
                        )
                        table_data.append({
                            "step": series_step,
                            "timestamp": timestamp,
                            "value": row["string_value"]
                        })
                if table_data:
                    table = wandb.Table(dataframe=pd.DataFrame(table_data))
                    self._active_run.log({attr_name: table})
            else:
                # Original: create text artifact
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", encoding="utf-8", delete=False
                ) as tmp_file:
                    for _, row in group.iterrows():
                        if pd.notna(row["string_value"]):
                            series_step = (
                                self._convert_step_to_int(row["step"], step_multiplier)
                                if pd.notna(row["step"])
                                else None
                            )
                            timestamp = (
                                row["timestamp"].isoformat()
                                if pd.notna(row["timestamp"])
                                else None
                            )
                            text_line = (
                                f"{series_step}; {timestamp}; {row['string_value']}\n"
                            )
                            tmp_file.write(text_line)
                    tmp_file_path = tmp_file.name

                artifact = wandb.Artifact(name=attr_name, type="string_series")
                artifact.add_file(tmp_file_path, name="series.txt")
                self._active_run.log_artifact(artifact)
                os.unlink(tmp_file_path)

        # Handle histogram series as W&B Histograms
        histogram_series_data = run_data[
            run_data["attribute_type"] == "histogram_series"
        ]
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)
            # Use global step multiplier

            for _, row in group.iterrows():
                if pd.notna(row["histogram_value"]) and isinstance(
                    row["histogram_value"], dict
                ):
                    step = (
                        self._convert_step_to_int(row["step"], step_multiplier)
                        if pd.notna(row["step"])
                        else 0
                    )
                    hist = row["histogram_value"]

                    # Convert Neptune histogram to W&B Histogram
                    # Neptune format: {"type": str, "edges": list, "values": list}
                    # W&B expects histogram data as np_histogram tuple or sequence
                    try:
                        wandb_hist = wandb.Histogram(
                            np_histogram=(hist.get("values", []), hist.get("edges", []))
                        )
                        self._active_run.log({attr_name: wandb_hist}, step=step)
                    except Exception:
                        self._logger.error(
                            f"Failed to log histogram for {attr_path} at step {step}",
                            exc_info=True,
                        )

        self._logger.info(f"Uploaded artifacts for run {run_id}")