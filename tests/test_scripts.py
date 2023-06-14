import pytest
import runpy
from click.testing import CliRunner
from run_radas import run_radas

@pytest.mark.slow
def test_fetch_adas_data(adas_data_directory):
    runpy.run_path(str(adas_data_directory / "fetch_adas_data.py"), run_name='__main__')

@pytest.mark.slow
def test_run_radas():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(run_radas, ['tungsten'])
    assert result.exit_code == 0
