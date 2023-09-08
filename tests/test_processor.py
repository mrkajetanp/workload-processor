import tarfile
import pytest
import confuse
import lisa
import wp
from pathlib import Path

from wp.processor import WorkloadProcessor
from wp.exception import WorkloadProcessorError, WPConfigError

TEST_OUTPUT_ARCHIVE = 'speedometer_test_minimal_2_0809.tar.gz'
TEST_OUTPUT_NAME = Path(Path(TEST_OUTPUT_ARCHIVE).stem).stem
TEST_PLATFORM_INFO = 'p6-platform-info.yml'

def prepare_test_output(resource_path_root, tmp_path):
    outgz = tarfile.open(resource_path_root / TEST_OUTPUT_ARCHIVE)
    outgz.extractall(tmp_path)
    outgz.close()
    return Path(tmp_path) / TEST_OUTPUT_NAME

def prepare_config(resource_path_root):
    config = confuse.Configuration('test_wp', __name__)
    config['target']['plat_info'].set(str(resource_path_root / TEST_PLATFORM_INFO))
    return config

def test_requires_output():
    with pytest.raises(FileNotFoundError):
        WorkloadProcessor('nonexistent')

def test_requires_jobs(tmp_path):
    with pytest.raises(WorkloadProcessorError):
        WorkloadProcessor(tmp_path)

def test_requires_plat_info(resource_path_root, tmp_path):
    output_path = prepare_test_output(resource_path_root, tmp_path)
    with pytest.raises(WPConfigError):
        WorkloadProcessor(output_path)

def test_initialises_correctly(resource_path_root, tmp_path):
    output_path = prepare_test_output(resource_path_root, tmp_path)
    config = prepare_config(resource_path_root)

    processor = WorkloadProcessor(output_path, config=config)
    assert processor.label == 'speedometer'
    assert len(processor.traces) == 2
    assert isinstance(processor.traces[1], lisa.trace.Trace)
    assert isinstance(processor.wa_output, lisa.wa.WAOutput)
    assert isinstance(processor.analysis, wp.analysis.WorkloadAnalysisRunner)
